import tensorflow as tf
import arch as ar 
import ops


class WaveNetTrain(ar.WaveNetArch):

    def __init__(self, sess, **kwargs):
        super().__init__(sess, **kwargs)

        self.l2_factor = kwargs['l2_factor']
        self.batch_sz = kwargs['batch_sz']
        
    def get_recep_field_sz(self):
        return self.n_blocks * sum([2**l for l in range(self.n_block_layers)])

    def encode_input_onehot(self, wav_input):
        '''
        wav_input[b][t], batch b, time t
        wav_input_onehot[b][t][q] for batch b, time t, quant channel q
        '''
        with tf.name_scope('encode'):
            if self.wav_input_type == 'mu_law_quant':
                wav_input_mu = wav_input
            elif self.wav_input_type == 'raw':
                wav_input_mu = ops.mu_encode(wav_input, self.n_quant)

            wav_input_onehot = tf.one_hot(wav_input_mu, self.n_quant, axis = -1,
                    name = 'one_hot_input')
        return wav_input_onehot

    def _preprocess_lc(self, lc_input):
        '''lc_input: batch x time x channel
        filt: stride '''
        lc_cur = lc_input 
        batch_ten = tf.constant(self.batch_sz)
        lc_out_ten = tf.constant(self.n_lc_out)

        for i, s in enumerate(self.lc_upsample):
            with tf.variable_scope('tconv{}'.format(i)):
                filt = self.get_variable(ar.ArchCat.LC_UPSAMPLE, i)
                slice_ten = tf.shape(lc_cur)[1] * s

                out_shape = tf.stack([batch_ten, slice_ten, lc_out_ten], axis=0)
                #out_shape = tf.constant([self.batch_sz, s, self.n_lc_out])
                lc_cur = tf.contrib.nn.conv1d_transpose(lc_cur, filt, out_shape, s)
        return lc_cur


    def _preprocess(self, wav_input):
        '''entry point of data coming from data.Dataset.
        wav_input: B x T x Q
        '''  
        # self.batch_sz = tf.shape(wav_input)[0]

        if self.has_global_cond():
            # gc_embeds[batch][t] = embedding vector
            self.gc_embeds = self.get_variable(ar.ArchCat.GC_EMBED)

        filt = self.get_variable(ar.ArchCat.PRE)
        trans = ops.conv1x1(wav_input, filt, self.batch_sz, 'conv')
        if self.use_bias:
            bias = self.get_variable(ar.ArchCat.PRE, get_bias=True)
            trans = tf.add(trans, bias)

        return trans


    def _map_embeds(self, id_mask, proj_filt, conv_name):
        '''create the batched, mapped, projected embedding.
        id_mask: B x T = gc_id
        gc_tab: batch x index x n_gc_embed 
        proj_embeds: batch x index x n_dil
        proj_filt: n_gc_embed x n_dil
        returns [batch, index, n_dil]'''
        gathered = tf.gather(self.gc_embeds, id_mask)
        proj_gathered = ops.conv1x1(gathered, proj_filt, self.batch_sz, conv_name)
        return proj_gathered

    
    def _dilated_conv(self, prev_z, lc_upsampled, dilation, id_mask, *var_indices): 
        '''construct one dilated, gated convolution as in equation 2 of WaveNet
        Sept 2016
        prev_z[b][t][r], for batch b, time t, res channel r
        '''
        # prev_z_save is already populated according to the current window
        prev_z_save = self.get_variable(ar.ArchCat.SAVE, dilation,
                *var_indices, trainable=False)

        # prev_z_rand = tf.random_uniform(saved_shape)
        prev_z_full = tf.concat([prev_z_save, prev_z], 1, name='concat')
        # prev_z_full = tf.concat([prev_z_save, prev_z], 1, name='concat')

        # construct signal and gate logic
        v = {}
        sig_gate = [ar.ArchCat.SIGNAL, ar.ArchCat.GATE]
        sig_gate_gc = [ar.ArchCat.GC_SIGNAL, ar.ArchCat.GC_GATE]
        sig_gate_lc = [ar.ArchCat.LC_SIGNAL, ar.ArchCat.LC_GATE]

        for arch in sig_gate:
            filt = self.get_variable(arch, *var_indices)
            #abs_mean = tf.reduce_mean(tf.abs(filt))
            #filt = tf.Print(filt, [abs_mean], 'D%i: ' % dilation)
            #prev_z_full = tf.Print(prev_z_full,
            #        [tf.shape(prev_z_full), tf.shape(filt)],
            #        'prev_z_full.shape, filt.shape: ')
            v[arch] = tf.nn.convolution(prev_z_full, filt, 'VALID',
                    [1], [dilation], 'conv_{}'.format(arch.name) )
            if self.use_bias:
                bias = self.get_variable(arch, *var_indices, get_bias=True)
                # bias = tf.Print(bias, [bias])
                v[arch] = tf.add(v[arch], bias, 'add_bias')

        if self.has_global_cond():
            for a, g in zip(sig_gate, sig_gate_gc): 
                gc_filt = self.get_variable(g, *var_indices)
                gc_proj = self._map_embeds(id_mask, gc_filt, 'gc_proj_embed')
                v[a] = tf.add(v[a], gc_proj, 'add')

        if lc_upsampled is not None:
            for a, l in zip(sig_gate, sig_gate_lc): 
                lc_filt = self.get_variable(l, *var_indices)
                lc_proj = ops.conv1x1(lc_upsampled, lc_filt, self.batch_sz, 'lc')
                v[a] = tf.add(v[a], lc_proj, 'add')

        
        # ensure we update prev_z_save before moving on
        # should this have a tf.stop_gradient?  what would that mean?
        aop = tf.assign(prev_z_save, prev_z_full[:,-dilation:,:])
        with tf.control_dependencies([aop]):
            z = tf.tanh(v[ar.ArchCat.SIGNAL]) * tf.sigmoid(v[ar.ArchCat.GATE])
        return z 


    def _chan_reduce(self, prev_op, *var_indices):
        chan = [ar.ArchCat.RESIDUAL, ar.ArchCat.SKIP]
        v = {}
        for arch in chan:
            filt = self.get_variable(arch, *var_indices)
            v[arch] = ops.conv1x1(prev_op, filt, self.batch_sz, 'conv_{}'.format(arch.name)) 
            #v[arch] = tf.nn.convolution(prev_op, filt, 'VALID', [1], [1],
            #        'conv_{}'.format(arch.name))
            if self.use_bias:
                bias = self.get_variable(arch, *var_indices, get_bias=True)
                v[arch] = tf.add(v[arch], bias, 'add_bias')

        signal, skip = v[chan[0]], v[chan[1]]
        return signal, skip 


    def _postprocess(self, input):
        '''implement the post-processing, just after the '+' sign and
        before the 'ReLU', where all skip connections add together.
        see section 2.4
        
        '''
        with tf.name_scope('postprocess'):
            relu1 = tf.nn.relu(input, 'ReLU')
            with tf.name_scope('chan'):
                post1_filt = self.get_variable(ar.ArchCat.POST1)
                dense1 = ops.conv1x1(relu1, post1_filt, self.batch_sz, 'conv') 
                #dense1 = tf.nn.convolution(relu1, post1_filt, 'VALID', [1], [1], 'conv')
                if self.use_bias:
                    bias = self.get_variable(ar.ArchCat.POST1, get_bias=True)
                    dense1 = tf.add(dense1, bias, 'add_bias')

            relu2 = tf.nn.relu(dense1, 'ReLU')
            with tf.name_scope('chan'):
                post2_filt = self.get_variable(ar.ArchCat.POST2)
                dense2 = ops.conv1x1(relu2, post2_filt, self.batch_sz, 'conv')
                # dense2 = tf.nn.convolution(relu2, post2_filt, 'VALID', [1], [1], 'conv')
                if self.use_bias:
                    bias = self.get_variable(ar.ArchCat.POST2, get_bias=True)
                    dense2 = tf.add(dense2, bias, 'add_bias')

            with tf.name_scope('softmax'):
                softmax = tf.nn.softmax(dense2, axis=2, name='softmax')

        return dense2, softmax


    def _loss_fcn(self, input, logits_out, id_mask, l2_factor):
        '''calculates cross-entropy loss with l2 regularization'''
        with tf.name_scope('loss'):
            shift_input = tf.stop_gradient(input[:,1:,:])
            logits_out_clip = logits_out[:,:-1,:]
            
            cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels = shift_input,
                    logits = logits_out_clip,
                    dim=2)
            #id_mask = tf.stack(id_masks)
            use_mask = tf.cast(tf.not_equal(id_mask[:,1:], 0), tf.float32)
            cross_ent_filt = cross_ent * use_mask 
            # cross_ent_filt = cross_ent
            mean_cross_ent = tf.reduce_mean(cross_ent_filt)
            with tf.name_scope('regularization'):
                if l2_factor != 0:
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)
                        for k, v in self.trainable_vars.items() if not 'BIAS' in k])
                else:
                    l2_loss = 0
            mean_cross_ent = tf.Print(mean_cross_ent,
                    [mean_cross_ent, l2_loss], 'MeanXEnt, L2: ')
            total_loss = mean_cross_ent + l2_factor * l2_loss

        return total_loss


    def build_graph(self, wav_input, lc_input, id_mask):
        '''creates the training graph and returns the loss node for the graph.
        the inputs to this function are data.Dataset.iterator.get_next() operations.
        This graph performs the forward calculation in parallel across
        a slice of time steps.  The loss compares these calculations with
        the next input value.
        wav_input: list of batch_sz wav_slices
        id_mask: B x T mask values (gc_id or 0, where 0 means invalid) 
        '''

        encoded_input = self.encode_input_onehot(wav_input)
        encoded_input = tf.stop_gradient(encoded_input)

        with tf.variable_scope('preprocess'):
            cur = self._preprocess(encoded_input)

        if self.use_lc_input():
            lc_upsampled = self._preprocess_lc(lc_input)
        else:
            lc_upsampled = None

        for b in range(self.n_blocks):
            with tf.variable_scope('block{}'.format(b)):
                for bl in range(self.n_block_layers):
                    with tf.variable_scope('layer{}'.format(bl)):
                        l = b * self.n_block_layers + bl
                        dil = 2**bl
                        dconv = self._dilated_conv(cur, lc_upsampled, dil, id_mask, b, bl)
                        sig, skp = self._chan_reduce(dconv, b, bl)
                        if b == 0 and bl == 0:
                            skp_sum = skp
                        else:
                            skp_sum = tf.add(skp_sum, skp, name='add_skip')
                        #skps.append(skp)
                        cur = tf.add(cur, sig, name='residual_add') 

        #skp_all = tf.add_n(skps, name='add_skip')
        logits, softmax_out = self._postprocess(skp_sum)
        # logits = self._postprocess(skp_all)
        loss = self._loss_fcn(encoded_input, logits, id_mask, self.l2_factor)

        diffs = tf.argmax(encoded_input[:,1:,:], 2) - tf.argmax(softmax_out[:,:-1,:], 2)
        absdiff = tf.abs(diffs)
        avg = tf.reduce_mean(absdiff)
        sum_valid = tf.reduce_sum(id_mask)

        loss = tf.Print(loss, [sum_valid, avg], 'Sum_Masks, Avg_ArgMaxAbsDiff: ', summarize=130)

        self.graph_built = True

        return loss 

    def grad_var_loss_eager(self, wav_input, lc_input, id_mask):
        with tf.GradientTape() as tape:
            loss = self.build_graph(wav_input, lc_input, id_mask)
            var_list = list(self.trainable_vars.values())
        grads = tape.gradient(loss, var_list)
        grads_vars = list(zip(grads, var_list))
        return grads_vars, loss


    def grad_var_loss(self, wav_input, lc_input, id_mask):
        loss_op = self.build_graph(wav_input, lc_input, id_mask)
        var_list = list(self.trainable_vars.values())
        opt = tf.train.Optimizer(True, 'lb-wavenet')
        grads_vars_op = opt.compute_gradients(loss, var_list) 
        return grads_vars_op, loss_op

