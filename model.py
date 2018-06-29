import tensorflow as tf


def mu_encode(x, n_quanta):
    '''mu-law encode and quantize'''
    mu = tf.to_float(n_quanta - 1)
    amp = tf.sign(x) * tf.log1p(mu * tf.abs(x)) / tf.log1p(mu)
    quant = (amp + 1) * 0.5 * mu + 0.5
    return tf.to_int32(quant)



def create_var(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable



class WaveNet(object):

    def __init__(self,
            n_blocks,
            n_block_layers,
            n_quant_chan,
            n_res_chan,
            n_dil_chan,
            n_skip_chan,
            n_post1_chan,
            n_gc_embed_chan,
            n_gc_category,
            l2_factor):

        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_quant_chan = n_quant_chan
        self.n_res_chan = n_res_chan
        self.n_dil_chan = n_dil_chan
        self.n_skip_chan = n_skip_chan
        self.n_post1_chan = n_post1_chan
        self.n_gc_embed_chan = n_gc_embed_chan
        self.n_gc_category = n_gc_category
        self.use_gc = n_gc_embed_chan > 0
        self.l2_factor = l2_factor
        self.filters = []
        self.saved = []
        
    def get_recep_field_sz(self):
        return self.n_blocks * sum([2**l for l in range(self.n_block_layers)])

    def encode_input_onehot(self, wav_input):
        with tf.name_scope('encode'):
            wav_input_mu = mu_encode(wav_input, self.n_quant_chan)
            # wav_input_onehot[b][t][i], batch b, time t, category i
            wav_input_onehot = tf.one_hot(wav_input_mu, self.n_quant_chan, axis = -1,
                    name = 'one_hot_input')
        return wav_input_onehot

    def _preprocess(self, wav_input_encoded, id_maps):
        '''entry point of data coming from data.Dataset.
        wav_input[b][t] for batch b, time t'''  
        filter_shape = [1, self.n_quant_chan, self.n_res_chan]

        with tf.name_scope('config_inputs'):
            self.batch_sz = tf.shape(wav_input_encoded)[0]

        with tf.name_scope('preprocess'):
            filt = create_var('in_filter', filter_shape)

            if self.use_gc:
                self.gc_tab = create_var('gc_tab',
                        [self.n_gc_category, self.n_gc_embed_chan])
                self.filters.append(self.gc_tab)
                # gc_embeds[batch][i] = embedding vector
                self.gc_embeds = [
                        tf.nn.embedding_lookup(self.gc_tab, m) for m in id_maps
                        ]
            pre_op = tf.nn.convolution(wav_input_encoded, filt,
                    'VALID', [1], [1], 'in_conv')

        self.filters.append(filt)
        return pre_op

    def _map_embeds(self, id_masks, proj_filt, conv_name):
        '''create the batched, mapped, projected embedding.
        returns shape [batch_sz, t, n_chan]'''
        proj_embeds = [
                tf.nn.convolution(tf.expand_dims(emb, 0),
                    proj_filt, 'VALID', [1], [1], conv_name)
                for emb in self.gc_embeds]
        gathered = [
                tf.gather(proj_emb[0], mask)
                for proj_emb, mask in zip(proj_embeds, id_masks)]
        return tf.stack(gathered)

    
    def _dilated_conv(self, prev_op, dilation, id_masks, batch_sz): 
        '''construct one dilated, gated convolution as in equation 2 of WaveNet Sept 2016'''
    
        # save last postiions from prev_op
        with tf.name_scope('load_store_values'):
            saved_shape = [batch_sz, dilation, self.n_res_chan]
            save = tf.Variable(tf.zeros(saved_shape), name='saved_length%i' % dilation,
                    validate_shape = False,
                    trainable = False)
            stop_grad = tf.stop_gradient(save)
            concat = tf.concat([stop_grad, prev_op], 1, name='concat')
            assign = save.assign(concat[:,-dilation:,:])
        self.saved.append(save)

        # construct signal and gate logic
        v = {}
        gc = {}
        for part in ['signal', 'gate']:
            conv_shape = [2, self.n_res_chan, self.n_dil_chan]
            with tf.name_scope(part):
                filt = create_var('filter', conv_shape) 
                self.filters.append(filt)
                v[part] = tf.nn.convolution(concat, filt, 
                        'VALID', [1], [dilation], 'dilation%i_conv' % dilation)

                if self.use_gc:
                    gc_conv_shape = [1, self.n_gc_embed_chan, self.n_dil_chan]
                    gc_filt = create_var('filter', gc_conv_shape) 
                    self.filters.append(gc_filt)
                    gc[part] = self._map_embeds(id_masks, gc_filt, 'gc_proj_embed')
        
        (signal, gate) = (v['signal'], v['gate'])
        if self.use_gc:
            signal = tf.add(signal, gc['signal'], 'add_gc_signal')
            gate = tf.add(gate, gc['gate'], 'add_gc_gate')

        post_signal = tf.tanh(signal)
        post_gate = tf.sigmoid(gate)

        with tf.control_dependencies([assign]):
            z = post_signal * post_gate
                
        return z 


    def _chan_reduce(self, prev_op):
        sig_shape = [1, self.n_dil_chan, self.n_res_chan]
        skip_shape = [1, self.n_dil_chan, self.n_skip_chan]
        with tf.name_scope('signal_%i_to_%i' % (sig_shape[1], sig_shape[2])):
            sig_filt = create_var('filter', sig_shape)
            self.filters.append(sig_filt)
            signal = tf.nn.convolution(prev_op, sig_filt,
                    'VALID', [1], [1], 'conv')

        with tf.name_scope('skip_%i_to_%i' % (skip_shape[1], skip_shape[2])):
            skip_filt = create_var('filter', skip_shape)
            self.filters.append(skip_filt)
            skip = tf.nn.convolution(prev_op, skip_filt,
                    'VALID', [1], [1], 'conv')

        return signal, skip 

    def _postprocess(self, prev_op):
        '''implement the post-processing, just after the '+' sign and
        before the 'ReLU', where all skip connections add together.
        see section 2.4'''
        shape1 = [1, self.n_skip_chan, self.n_post1_chan]
        shape2 = [1, self.n_post1_chan, self.n_quant_chan]
        with tf.name_scope('postprocess'):
            relu1 = tf.nn.relu(prev_op, 'ReLU')
            with tf.name_scope('chan_%i_to_%i' % (shape1[1], shape1[2])):
                filt1 = create_var('filter', shape1)
                dense1 = tf.nn.convolution(relu1, filt1, 'VALID', [1], [1], 'conv')

            relu2 = tf.nn.relu(dense1, 'ReLU')
            with tf.name_scope('chan_%i_to_%i' % (shape2[1], shape2[2])):
                filt2 = create_var('filter', shape2)
                dense2 = tf.nn.convolution(relu2, filt2, 'VALID', [1], [1], 'conv')

            with tf.name_scope('softmax'):
                softmax = tf.nn.softmax(dense2, 0, 'softmax')

        self.filters.append(filt1)
        self.filters.append(filt2)
        return dense2, softmax


    def _loss_fcn(self, wav_input_encoded, id_masks, net_logits_out, l2_factor):
        '''calculates cross-entropy loss with l2 regularization'''
        with tf.name_scope('loss'):
            shift_input = wav_input_encoded[:,1:,:]
            cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels = shift_input,
                    logits = net_logits_out)
            id_mask = tf.stack(id_masks)
            use_mask = tf.cast(tf.not_equal(id_mask[:,1:], 0), tf.float32)
            cross_ent_filt = cross_ent * use_mask 
            mean_cross_ent = tf.reduce_mean(cross_ent_filt)
            with tf.name_scope('regularization'):
                if l2_factor != 0:
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                        for v in tf.trainable_variables()
                        if not ('bias' in v.name)])
                else:
                    l2_loss = 0
            total_loss = mean_cross_ent + l2_factor * l2_loss

        return total_loss

    def initialize_training_graph(self, sess):
        sess.run([
            tf.variables_initializer(self.saved),
            tf.variables_initializer(self.filters)
            ])



    def create_training_graph(self, wav_input, id_masks, id_maps):
        '''creates the training graph and returns the loss node for the graph.
        the inputs to this function are data.Dataset.iterator.get_next() operations.
        This graph performs the forward calculation in parallel across
        a slice of time steps.  The loss compares these calculations with
        the next input value.'''

        # use the same graph as the input
        graph = wav_input.graph 
        with graph.as_default():
            encoded_input = self.encode_input_onehot(wav_input)
            cur = self._preprocess(encoded_input, id_maps)
            skip = []

            for b in range(self.n_blocks):
                with tf.name_scope('block%i' % (b + 1)):
                    for l in range(self.n_block_layers):
                        with tf.name_scope('layer%i' % (l + 1)):
                            dil = 2**l
                            dil_conv_op = self._dilated_conv(cur, dil,
                                    id_masks, self.batch_sz)
                            (signal_op, skip_op) = self._chan_reduce(dil_conv_op)
                            skip.append(skip_op)
                            cur = tf.add(cur, signal_op, name='residual_add') 

            sum_all = sum(skip)
            (logits, softmax_out) = self._postprocess(sum_all)
            loss = self._loss_fcn(encoded_input, id_masks, logits, self.l2_factor)

        return loss 

