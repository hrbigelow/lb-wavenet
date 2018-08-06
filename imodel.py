import tensorflow as tf
import arch as ar
import ops
from sys import stderr


class WaveNetGen(ar.WaveNetArch):

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
            batch_sz,
            chunk_sz,
            teacher_vec
            ):

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
                add_summary=False)

        self.chunk_sz = chunk_sz
        self.batch_sz = batch_sz
        self.single_assign = []
        self.shift_assign = []
        self.lookback_buffers = []
        self.loop_buffers = []
        if teacher_vec is not None:
            self.teacher_vec = teacher_vec
            self.teacher_mu = ops.mu_encode(self.teacher_vec, self.n_quant)
            print('Teacher vec is {} samples long.'.format(self.teacher_vec.shape[0]),
                file=stderr)


    def _non_loop_init(self):
        '''operations that will run before the loop starts'''
        if self.use_gc:
            self.gc_ids = tf.placeholder(tf.int32, (self.batch_sz,), 'gc_ids')
            gc_tab = self.get_variable(ar.ArchCat.GC_EMBED)
            self.gc_embeds = tf.gather(gc_tab, self.gc_ids)
            
        self.gen_sz = tf.placeholder(tf.int32, (), 'gen_sz')


    def _preprocess(self, gpos):
        '''
        everything done to the raw signal before it is ready for the layers.
        src_buf: [batch_sz, chunk_sz, n_quant]
        '''  
        one_hot = tf.get_variable('input',
                [self.batch_sz, self.chunk_sz, self.n_quant],
                initializer=tf.zeros_initializer,
                trainable=False,
                collections=None)
        self.loop_buffers.append(one_hot)

        filt = self.get_variable(ar.ArchCat.PRE)
        z = tf.matmul(one_hot[:,gpos,:], filt)
        #if self.use_bias:
        #    bias = self.get_variable(ar.ArchCat.PRE, get_bias=True)
        #    z = tf.add(z, bias)

        return one_hot, z 


    def _dilated_conv(self, z_in, dilation, wpos, *var_indices):
        '''
        z_in: latest  
        wpos: zero-based position in the cached window
        prev_z: value from previous layer at time t 
        '''
        saved_shape = [self.batch_sz, dilation + self.chunk_sz, self.n_res]
        bias_init = self.get_variable
        z_in_saved = tf.get_variable(
                name='lookback_buffer',
                shape=saved_shape,
                initializer=tf.zeros_initializer,
                trainable=False)
        self.lookback_buffers.append(z_in_saved)

        aop = tf.assign(z_in_saved[:,wpos + dilation,:], z_in)
        self.single_assign.append(aop)

        v = {}
        sig_gate = [ar.ArchCat.SIGNAL, ar.ArchCat.GATE]
        sig_gate_gc = [ar.ArchCat.GC_SIGNAL, ar.ArchCat.GC_GATE]

        for arch in sig_gate:
            with tf.variable_scope('conv/{}'.format(arch.name)):
                filt = self.get_variable(arch, *var_indices)
                v[arch] = tf.matmul(z_in_saved[:,wpos,:], filt[0]) \
                        + tf.matmul(z_in, filt[1])
                if self.use_bias:
                    bias = self.get_variable(arch, *var_indices, get_bias=True)
                    v[arch] = tf.add(v[arch], bias, 'add_bias')

        if self.use_gc:
            for a, g in zip(sig_gate, sig_gate_gc): 
                with tf.variable_scope('gc/{}'.format(a.name)):
                    gc_filt = self.get_variable(g, *var_indices)
                    gc_proj = tf.matmul(self.gc_embeds, gc_filt)
                    v[a] = tf.add(v[a], gc_proj, 'add')

        with tf.variable_scope('gating'):
            dconv = tf.tanh(v[ar.ArchCat.SIGNAL]) * tf.sigmoid(v[ar.ArchCat.GATE])
        return dconv


    def _chan_reduce(self, prev_op, *var_indices):
        '''simply provide the channel reducing operations'''
        chan = [ar.ArchCat.RESIDUAL, ar.ArchCat.SKIP]
        v = {}
        for arch in chan:
            filt = self.get_variable(arch, *var_indices)
            v[arch]= tf.matmul(prev_op, filt)
            if self.use_bias:
                bias = self.get_variable(arch, *var_indices, get_bias=True)
                v[arch] = tf.add(v[arch], bias, 'add_bias')

        signal, skip = v[chan[0]], v[chan[1]]
        return signal, skip 


    def _postprocess(self, input):
        '''implement post-processing, just after the '+' sign and
        before the 'ReLU'.  See section 2.4
        input: B x S 
        '''
        relu1 = tf.nn.relu(input, 'ReLU')
        with tf.name_scope('chan'):
            post1_filt = self.get_variable(ar.ArchCat.POST1)
            dense1 = tf.matmul(relu1, post1_filt)
            if self.use_bias:
                bias = self.get_variable(ar.ArchCat.POST1, get_bias=True)
                dense1 = tf.add(dense1, bias, 'add_bias')

        relu2 = tf.nn.relu(dense1, 'ReLU')
        with tf.name_scope('chan'):
            post2_filt = self.get_variable(ar.ArchCat.POST2)
            dense2 = tf.matmul(relu2, post2_filt) 
            if self.use_bias:
                bias = self.get_variable(ar.ArchCat.POST2, get_bias=True)
                dense2 = tf.add(dense2, bias, 'add_bias')

        #with tf.name_scope('softmax'):
        #    softmax = tf.nn.softmax(dense2, 0, 'softmax')

        return dense2 #, softmax


    def _sample_next(self, logits, wpos):
        '''obtain a random sample from the most recent logits output,
        one-hot encode it, and populate the next input element
        logits: B x C 
        wpos: current window position
        '''
        wav_buf = tf.get_variable('output',
                [self.batch_sz, self.chunk_sz],
                initializer=tf.zeros_initializer,
                trainable=False)
        self.loop_buffers.append(wav_buf)

        samp = tf.squeeze(tf.multinomial(logits, 1), 1)
        hot = tf.one_hot(samp, self.n_quant)
        wav_val = ops.mu_decode(samp, self.n_quant)
        aop = tf.assign(wav_buf[:,wpos], wav_val)
        aop = tf.Print(aop, [wpos, samp, wav_val], 'wpos, samp, wav_val', summarize=20)
        self.single_assign.append(aop)
        # aop = tf.Print(aop, [tf.argmax(logits, 1)], 'max_logits_index')

        return hot, wav_buf 


    def _next_window(self, wav_buf, wav):
        '''
        called after each chunk_sz iterations of _loop_body.
        appends the populated window to the final output buffer,
        and advances all lookback buffers.
        wav_buf: [batch_sz, chunk_sz]
        wav: [batch_sz, total_length]
        '''
        # copy len-chunk_sz last values to beginning
        for v in self.lookback_buffers:
            op = tf.assign(v[:,:-self.chunk_sz,:], v[:,self.chunk_sz:,:])
            self.shift_assign.append(op)

        # wav_buf = tf.Print(wav_buf, [wav_buf], 'wav_buf = ')
        with tf.control_dependencies(self.shift_assign):
            wav = tf.concat([wav, wav_buf], 1) 
            wav = tf.Print(wav, [tf.shape(wav)[1]], 'Total samples: ')
        return wav 


    def _loop_cond(self, i, wav, wpos):
        return i < self.gen_sz 


    def _loop_body(self, i, wav, wpos):
        '''while loop body for inference
        wav: [batch_sz, total_length] final output wav data
        i: loop iteration
        wpos: window position in [0, chunk_sz)
        
        intermediate buffers used:
        in_buf: holds one-hot vectors which encode the sampled logits from t-1
        pre_buf: linearly transformed in_buf values
        dst_buf (one for each layer): holds the output of the convolutional layer
        wav_buf: holds the final output'''
        
        # invoking the variable scope allows _loop_body to scope these
        # variables within the while loop without re-scoping the operations
        name_scope = tf.get_default_graph().get_name_scope()
        with tf.variable_scope(name_scope, auxiliary_name_scope=False):

            with tf.variable_scope('preprocess'):
                # in_buf has one-hot vectors of size self.n_quant
                one_hot, z = self._preprocess(wpos)

            skps = []
            for b in range(self.n_blocks):
                with tf.variable_scope('block{}'.format(b)):
                    for bl in range(self.n_block_layers):
                        with tf.variable_scope('layer{}'.format(bl)):
                            l = b * self.n_block_layers + bl
                            dil = 2**bl
                            dconv = self._dilated_conv(z, dil, wpos, b, bl)
                            sig, skp = self._chan_reduce(dconv, b, bl)
                            skps.append(skp)
                            z = tf.add(z, sig, name='residual_add')

            skp_all = tf.add_n(skps, name='add_skip')
            with tf.variable_scope('postprocess'):
                #logits, softmax_out = self._postprocess(skp_all)
                logits = self._postprocess(skp_all)
                hot, wav_buf = self._sample_next(logits, wpos)

            # hot = tf.Print(hot, [tf.argmax(hot, axis=1)], 'hot = ')

            with tf.control_dependencies(self.single_assign):
                wav_nxt, wpos_nxt = tf.cond(tf.equal(wpos + 1, self.chunk_sz),
                        lambda: (self._next_window(wav_buf, wav), 0),
                        lambda: (wav, wpos + 1))

            if self.teacher_vec is not None:
                def get_teacher(i):
                    return tf.broadcast_to(tf.one_hot(self.teacher_mu[i], self.n_quant),
                            [self.batch_sz, self.n_quant])

                hot = tf.cond(tf.less(i, self.teacher_mu.shape[0]),
                        lambda: get_teacher(i),
                        lambda: hot)

            aop = tf.assign(one_hot[:,wpos_nxt], hot)
            with tf.control_dependencies([aop]):
                i = tf.identity(i)
            return i + 1, wav_nxt, wpos_nxt

    def init_buffers(self, sess):
        '''initialize all non-trainable variable buffers used during inference'''
        sess.run(self.buffers_init)


    def build_graph(self):
        '''create the whole inference graph, which is capable of generating some
        number of waveforms incrementally in parallel'''
        with tf.variable_scope('preprocess'): 
            self._non_loop_init()

        seed = tf.zeros([self.batch_sz, 0])
        zero_d = tf.TensorShape(())
        two_d = tf.TensorShape([self.batch_sz, None])

        i, waveform, wpos = tf.while_loop(
                self._loop_cond,
                self._loop_body,
                loop_vars=(0, seed, 0),
                shape_invariants=(zero_d, two_d, zero_d),
                parallel_iterations=1,
                back_prop=False,
                name='main_loop'
                )

        self.buffers_init = tf.variables_initializer(
                self.loop_buffers + self.lookback_buffers)

        self.graph_built = True
        return i, waveform, wpos



