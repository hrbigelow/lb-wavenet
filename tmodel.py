import tensorflow as tf
import arch as ar 
import ops


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
            use_bias,
            l2_factor,
            batch_sz,
            add_summary):

        super().__init__(
                n_blocks,
                n_block_layers,
                n_quant,
                n_res,
                n_dil,
                n_skip,
                n_post1,
                n_gc_embed,
                n_gc_category,
                use_bias,
                add_summary)

        self.l2_factor = l2_factor
        self.batch_sz = batch_sz
        
    def get_recep_field_sz(self):
        return self.n_blocks * sum([2**l for l in range(self.n_block_layers)])

    def encode_input_onehot(self, wav_input):
        '''
        wav_input[b][t], batch b, time t
        wav_input_onehot[b][t][q] for batch b, time t, quant channel q
        '''
        with tf.name_scope('encode'):
            wav_input_mu = ops.mu_encode(wav_input, self.n_quant)
            wav_input_onehot = tf.one_hot(wav_input_mu, self.n_quant, axis = -1,
                    name = 'one_hot_input')
        return wav_input_onehot

    def _preprocess(self, wav_input_encoded, id_maps):
        '''entry point of data coming from data.Dataset.
        wav_input_encoded[b][t][q] for batch b, time t, quant channel q'''  
        # self.batch_sz = tf.shape(wav_input_encoded)[0]

        if self.use_gc:
            # gc_embeds[batch][i] = embedding vector
            gc_tab = self.get_variable(ar.ArchCat.GC_EMBED)
            self.gc_embeds = [tf.nn.embedding_lookup(gc_tab, m) for m in id_maps]

        filt = self.get_variable(ar.ArchCat.PRE)
        pre_op = tf.nn.convolution(wav_input_encoded, filt,
                'VALID', [1], [1], 'conv')
        return pre_op

    def _map_embeds(self, id_masks, proj_filt, conv_name):
        '''create the batched, mapped, projected embedding.
        id_masks: [batch_sz, t] = gc_id
        proj_filt: [1, n_gc_embed, n_dil]
        self.gc_embeds: 
        returns shape [batch_sz, t, n_chan]'''
        proj_embeds = [
                tf.nn.convolution(tf.expand_dims(emb, 0),
                    proj_filt, 'VALID', [1], [1], conv_name)
                for emb in self.gc_embeds]
        gathered = [
                tf.gather(proj_emb[0], mask)
                for proj_emb, mask in zip(proj_embeds, id_masks)]
        return tf.stack(gathered)

    
    def _dilated_conv(self, prev_z, dilation, id_masks): 
        '''construct one dilated, gated convolution as in equation 2 of WaveNet
        Sept 2016
        prev_z[b][t][r], for batch b, time t, res channel r
        '''
    
        # prev_z_save is already populated according to the current window
        saved_shape = [self.batch_sz, dilation, self.n_res]
        prev_z_save = tf.get_variable(
                name='lookback_buffer',
                shape=saved_shape,
                initializer=tf.zeros_initializer,
                trainable=False
                )
        prev_z_rand = tf.random_uniform(saved_shape)
        prev_z_full = tf.concat([prev_z_rand, prev_z], 1, name='concat')

        # prev_z_full = tf.concat([prev_z_save, prev_z], 1, name='concat')

        # construct signal and gate logic
        v = {}
        sig_gate = [ar.ArchCat.SIGNAL, ar.ArchCat.GATE]
        sig_gate_gc = [ar.ArchCat.GC_SIGNAL, ar.ArchCat.GC_GATE]

        for arch in sig_gate:
            filt = self.get_variable(arch)
            #abs_mean = tf.reduce_mean(tf.abs(filt))
            #filt = tf.Print(filt, [abs_mean], 'D%i: ' % dilation)
            v[arch] = tf.nn.convolution(prev_z_full, filt, 'VALID',
                    [1], [dilation], 'conv')
            if self.use_bias:
                bias = self.get_variable(arch, True)
                # bias = tf.Print(bias, [bias])
                v[arch] = tf.add(v[arch], bias, 'add_bias')

        if self.use_gc:
            for a, g in zip(sig_gate, sig_gate_gc): 
                gc_filt = self.get_variable(g)
                gc_proj = self._map_embeds(id_masks, gc_filt, 'gc_proj_embed')
                v[a] = tf.add(v[a], gc_proj, 'add')
        
        # ensure we update prev_z_save before movign on
        #aop = tf.assign(prev_z_save, prev_z[:,-dilation:,:])
        #with tf.control_dependencies([aop]):
        #    z = tf.tanh(v[ar.ArchCat.SIGNAL]) * tf.sigmoid(v[ar.ArchCat.GATE])
                
        z = tf.tanh(v[ar.ArchCat.SIGNAL]) * tf.sigmoid(v[ar.ArchCat.GATE])
        return z 


    def _chan_reduce(self, prev_op):
        chan = [ar.ArchCat.RESIDUAL, ar.ArchCat.SKIP]
        v = {}
        for arch in chan:
            filt = self.get_variable(arch)
            v[arch] = tf.nn.convolution(prev_op, filt, 'VALID', [1], [1], 'conv')
            if self.use_bias:
                bias = self.get_variable(arch, True)
                v[arch] = tf.add(v[arch], bias, 'add_bias')

        signal, skip = v[chan[0]], v[chan[1]]
        return signal, skip 


    def _postprocess(self, prev_op):
        '''implement the post-processing, just after the '+' sign and
        before the 'ReLU', where all skip connections add together.
        see section 2.4'''
        post1_filt = self.get_variable(ar.ArchCat.POST1)
        post2_filt = self.get_variable(ar.ArchCat.POST2)
        with tf.name_scope('postprocess'):
            relu1 = tf.nn.relu(prev_op, 'ReLU')
            with tf.name_scope('chan'):
                dense1 = tf.nn.convolution(relu1, post1_filt, 'VALID', [1], [1], 'conv')
                if self.use_bias:
                    bias = self.get_variable(ar.ArchCat.POST1, True)
                    dense1 = tf.add(dense1, bias, 'add_bias')

            relu2 = tf.nn.relu(dense1, 'ReLU')
            with tf.name_scope('chan'):
                dense2 = tf.nn.convolution(relu2, post2_filt, 'VALID', [1], [1], 'conv')
                if self.use_bias:
                    bias = self.get_variable(ar.ArchCat.POST2, True)
                    dense2 = tf.add(dense2, bias, 'add_bias')

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
            # cross_ent_filt = cross_ent * use_mask 
            cross_ent_filt = cross_ent
            mean_cross_ent = tf.reduce_mean(cross_ent_filt)
            with tf.name_scope('regularization'):
                if l2_factor != 0:
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 
                        if not ('BIAS' in v.name)])
                else:
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                        for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 
                        if not ('BIAS' in v.name)])
                    # l2_loss = 0
            mean_cross_ent = tf.Print(mean_cross_ent, [mean_cross_ent, l2_loss], 'M,L: ')
            total_loss = mean_cross_ent + l2_factor * l2_loss

        return total_loss


    def build_graph(self, wav_input, id_masks, id_maps):
        '''creates the training graph and returns the loss node for the graph.
        the inputs to this function are data.Dataset.iterator.get_next() operations.
        This graph performs the forward calculation in parallel across
        a slice of time steps.  The loss compares these calculations with
        the next input value.
        wav_input: list of batch_sz wav_slices
        id_masks: list of batch_sz id_masks
        id_maps: list of batch_sz id_maps '''

        encoded_input = self.encode_input_onehot(wav_input)
        with tf.variable_scope('preprocess'):
            cur = self._preprocess(encoded_input, id_maps)
        skps = []

        for b in range(self.n_blocks):
            with tf.variable_scope('block{}'.format(b)):
                for bl in range(self.n_block_layers):
                    with tf.variable_scope('layer{}'.format(bl)):
                        l = b * self.n_block_layers + bl
                        dil = 2**bl
                        dconv = self._dilated_conv(cur, dil, id_masks)
                        sig, skp = self._chan_reduce(dconv)
                        skps.append(skp)
                        cur = tf.add(cur, sig, name='residual_add') 

        skp_all = sum(skps)
        logits, softmax_out = self._postprocess(skp_all)
        loss = self._loss_fcn(encoded_input, logits, id_masks, self.l2_factor)

        self.graph_built = True

        return loss 

