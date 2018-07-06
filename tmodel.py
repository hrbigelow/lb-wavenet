import tensorflow as tf
import arch as ar 


def mu_encode(x, n_quanta):
    '''mu-law encode and quantize'''
    mu = tf.to_float(n_quanta - 1)
    amp = tf.sign(x) * tf.log1p(mu * tf.abs(x)) / tf.log1p(mu)
    quant = (amp + 1) * 0.5 * mu + 0.5
    return tf.to_int32(quant)



class WaveNetTrain(ar.WaveNetArch):

    def __init__(self,
            n_blocks,
            n_block_layers,
            n_quant,
            n_res,
            n_dil,
            n_skip,
            n_post1,
            n_gc_embed,
            n_gc_category,
            l2_factor):

        super().__init__(
                n_blocks,
                n_block_layers,
                n_quant,
                n_res,
                n_dil,
                n_skip,
                n_post1,
                n_gc_embed,
                n_gc_category)

        self.l2_factor = l2_factor
        self.saved = []
        
    def get_recep_field_sz(self):
        return self.n_blocks * sum([2**l for l in range(self.n_block_layers)])

    def encode_input_onehot(self, wav_input):
        with tf.name_scope('encode'):
            wav_input_mu = mu_encode(wav_input, self.n_quant)
            # wav_input_onehot[b][t][i], batch b, time t, category i
            wav_input_onehot = tf.one_hot(wav_input_mu, self.n_quant, axis = -1,
                    name = 'one_hot_input')
        return wav_input_onehot

    def _preprocess(self, wav_input_encoded, id_maps):
        '''entry point of data coming from data.Dataset.
        wav_input[b][t] for batch b, time t'''  
        with tf.name_scope('config_inputs'):
            self.batch_sz = tf.shape(wav_input_encoded)[0]

        with tf.name_scope('preprocess'):
            if self.use_gc:
                # gc_embeds[batch][i] = embedding vector
                gc_tab = self.get_var('GE', ar.ArchCat.PRE)
                self.gc_embeds = [tf.nn.embedding_lookup(gc_tab, m) for m in id_maps]

            filt = self.get_var('QR', ar.ArchCat.PRE)
            pre_op = tf.nn.convolution(wav_input_encoded, filt,
                    'VALID', [1], [1], 'in_conv')
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

    
    def _dilated_conv(self, prev_op, dilation, block, layer, id_masks, batch_sz): 
        '''construct one dilated, gated convolution as in equation 2 of WaveNet Sept 2016'''
    
        # save last postiions from prev_op
        with tf.name_scope('load_store_values'):
            saved_shape = [batch_sz, dilation, self.n_res]
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
        for arch in [ar.ArchCat.SIGNAL, ar.ArchCat.GATE]:
            filt = self.get_var('RD', arch, block, layer)
            with tf.name_scope(arch.name):
                v[arch] = tf.nn.convolution(concat, filt, 
                        'VALID', [1], [dilation], 'dilation%i_conv' % dilation)

                if self.use_gc:
                    gc_filt = self.get_var('ED', arch, block, layer)
                    gc[arch] = self._map_embeds(id_masks, gc_filt, 'gc_proj_embed')
        
        (signal, gate) = (v[ar.ArchCat.SIGNAL], v[ar.ArchCat.GATE])
        if self.use_gc:
            signal = tf.add(signal, gc[ar.ArchCat.SIGNAL], 'add_gc_signal')
            gate = tf.add(gate, gc[ar.ArchCat.GATE], 'add_gc_gate')

        post_signal = tf.tanh(signal)
        post_gate = tf.sigmoid(gate)

        with tf.control_dependencies([assign]):
            z = post_signal * post_gate
                
        return z 


    def _chan_reduce(self, prev_op, block, layer):
        sig_filt = self.get_var('DR', ar.ArchCat.RESIDUAL, block, layer)
        skip_filt = self.get_var('DS', ar.ArchCat.SKIP, block, layer)
        with tf.name_scope('signal'):
            signal = tf.nn.convolution(prev_op, sig_filt, 'VALID', [1], [1], 'conv')
        with tf.name_scope('skip'):
            skip = tf.nn.convolution(prev_op, skip_filt, 'VALID', [1], [1], 'conv')
        return signal, skip 


    def _postprocess(self, prev_op):
        '''implement the post-processing, just after the '+' sign and
        before the 'ReLU', where all skip connections add together.
        see section 2.4'''
        post1_filt = self.get_var('SP', ar.ArchCat.POST)
        post2_filt = self.get_var('PQ', ar.ArchCat.POST)
        with tf.name_scope('postprocess'):
            relu1 = tf.nn.relu(prev_op, 'ReLU')
            with tf.name_scope('chan'):
                dense1 = tf.nn.convolution(relu1, post1_filt, 'VALID', [1], [1], 'conv')

            relu2 = tf.nn.relu(dense1, 'ReLU')
            with tf.name_scope('chan'):
                dense2 = tf.nn.convolution(relu2, post2_filt, 'VALID', [1], [1], 'conv')

            with tf.name_scope('softmax'):
                softmax = tf.nn.softmax(dense2, 0, 'softmax')

        return dense2, softmax


    def _loss_fcn(self, wav_input_encoded, logits_out, id_masks, l2_factor):
        '''calculates cross-entropy loss with l2 regularization'''
        with tf.name_scope('loss'):
            shift_input = wav_input_encoded[:,1:,:]
            logits_out_clip = logits_out[:,:-1,:]

            cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels = shift_input,
                    logits = logits_out_clip)
            id_mask = tf.stack(id_masks)
            use_mask = tf.cast(tf.not_equal(id_mask[:,1:], 0), tf.float32)
            cross_ent_filt = cross_ent * use_mask 
            mean_cross_ent = tf.reduce_mean(cross_ent_filt)
            # !!! fix the 'bias' in v.name test
            with tf.name_scope('regularization'):
                if l2_factor != 0:
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                        for v in self.vars.values() 
                        if not ('bias' in v.name)])
                else:
                    l2_loss = 0
            total_loss = mean_cross_ent + l2_factor * l2_loss

        return total_loss

    def initialize_training_graph(self, sess):
        sess.run([
            tf.variables_initializer(self.saved),
            self.var_init_op
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
                            dil_conv_op = self._dilated_conv(
                                    cur, dil, b, l,
                                    id_masks, self.batch_sz)
                            (signal_op, skip_op) = \
                                self._chan_reduce(dil_conv_op, b, l)
                            skip.append(skip_op)
                            cur = tf.add(cur, signal_op, name='residual_add') 

            sum_all = sum(skip)
            (logits, softmax_out) = self._postprocess(sum_all)
            loss = self._loss_fcn(encoded_input, logits, id_masks, self.l2_factor)

        return loss 

