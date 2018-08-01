import tensorflow as tf
import arch as ar
import ops


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
            chunk_sz):

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
        self.loop_buf_coll = 'loop_buffers'
        self.lb_buf_coll = 'lookback_buffers'

    def _non_loop_init(self):
        '''operations that will run before the loop starts'''
        if self.use_gc:
            self.gc_ids = tf.placeholder(tf.int32, (self.batch_sz,), 'gc_ids')
            gc_tab = self.get_variable(ar.ArchCat.GC_EMBED)
            self.gc_embeds = tf.gather(gc_tab, self.gc_ids)
            
        self.gen_sz = tf.placeholder(tf.int32, (), 'gen_sz')


    def _preprocess(self, src_buf, gpos):
        '''
        everything done to the raw signal before it is ready for the layers.
        src_buf: [batch_sz, chunk_sz, n_quant]
        '''  
        filt = self.get_variable(ar.ArchCat.PRE)
        pre_op = tf.matmul(src_buf[:,gpos,:], filt[0])

        return pre_op


    def _dilated_conv(self, prev_z, pos):
        '''
        pos: zero-based position in the cached window
        prev_z: value from previous layer at time t 
        '''
        saved_shape = [self.batch_sz, dil + self.chunk_sz, self.n_res]
        prev_z_save = tf.get_variable(
                name='lookback_buffer',
                shape=saved_shape,
                initializer=tf.zeros_initializer,
                trainable=False,
                collections=lookback_bufs)
        v = {}
        sig_gate = [ar.ArchCat.SIGNAL, ar.ArchCat.GATE]
        sig_gate_gc = [ar.ArchCat.GC_SIGNAL, ar.ArchCat.GC_GATE]

        for arch in sig_gate:
            filt = self.get_variable(arch)
            v[arch] = tf.matmul(prev_z_save[:,pos,:], filt[0]) \
            + tf.matmul(prev_z, filt[1])

        if self.use_gc:
            for a, g in zip(sig_gate, sig_gate_gc): 
                gc_filt = self.get_variable(g)
                gc_proj = tf.matmul(self.gc_embeds, gc_filt[0])
                v[a] = tf.add(v[a], gc_proj, 'add')

        z = tf.tanh(v[ar.ArchCat.SIGNAL]) * tf.sigmoid(v[ar.ArchCat.GATE])
        return z


    def _chan_reduce(self, prev_op):
        '''simply provide the channel reducing operations'''
        sig_filt = self.get_variable(ar.ArchCat.RESIDUAL)
        skp_filt = self.get_variable(ar.ArchCat.SKIP)
        signal = tf.matmul(prev_op, sig_filt[0], name='signal')
        skip = tf.matmul(prev_op, skp_filt[0], name='skip')

        return signal, skip 


    def _postprocess(self, prev_op):
        '''implement post-processing, just after the '+' sign and
        before the 'ReLU'.  See section 2.4'''
        post1_filt = self.get_variable(ar.ArchCat.POST1)
        post2_filt = self.get_variable(ar.ArchCat.POST2)

        with tf.name_scope('postprocess'):
            relu1 = tf.nn.relu(prev_op, 'ReLU')
            with tf.name_scope('chan'):
                dense1 = tf.matmul(relu1, post1_filt[0])

            relu2 = tf.nn.relu(dense1, 'ReLU')
            with tf.name_scope('chan'):
                dense2 = tf.matmul(relu2, post2_filt[0]) 

            with tf.name_scope('softmax'):
                softmax = tf.nn.softmax(dense2, 0, 'softmax')

        return dense2, softmax


    def _sample_next(self, logits):
        '''obtain a random sample from the most recent softmax output,
        one-hot encode it, and populate the next input element
        '''
        samp = tf.squeeze(tf.multinomial(logits, 1), 1)
        hot = tf.one_hot(samp, self.n_quant)
        wav_val = ops.mu_decode(samp, self.n_quant)
        return samp, hot, wav_val 

    def _next_window(self, wav_buf, wav):
        '''
        called after each chunk_sz iterations of _loop_body.
        appends the populated window to the final output buffer,
        and advances all lookback buffers.
        wav_buf: [batch_sz, chunk_sz]
        wav: [batch_sz, total_length]
        '''
        lb_vars = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                if 'lookback' in v.name]
        # copy len-chunk_sz last values to beginning
        aops = []
        for v in lb_vars:
            op = tf.assign(v[:,:-self.chunk_sz,:], v[:,self.chunk_sz:,:])
            aops.append(op)

        with tf.control_dependencies(aops):
            wav = tf.concat([wav, wav_buf], 1) 
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
        
        just_loop_bufs = [tf.GraphKeys.GLOBAL_VARIABLES, self.loop_buf_coll]
        lookback_bufs = just_loop_bufs + [self.lb_buf_coll]

        with tf.variable_scope('preprocess', reuse=None):
            # in_buf has one-hot vectors of size self.n_quant
            in_buf = tf.get_variable('input',
                    [self.batch_sz, self.chunk_sz, self.n_quant],
                    initializer=tf.zeros_initializer,
                    trainable=False,
                    collections=just_loop_bufs)
            pre_buf = tf.get_variable('input_trans',
                    [self.batch_sz, self.chunk_sz, self.n_res],
                    initializer=tf.zeros_initializer,
                    trainable=False,
                    collections=just_loop_bufs)
            cur = self._preprocess(in_buf, wpos)

        aop = tf.assign(pre_buf[wpos], cur)
        with tf.control_dependencies([aop]):
            cur_buf = pre_buf 

        skps = []
        for b in range(self.n_blocks):
            with tf.variable_scope('block{}'.format(b)):
                for bl in range(self.n_block_layers):
                    with tf.variable_scope('layer{}'.format(bl)):
                        l = b * self.n_block_layers + bl
                        dil = 2**bl
                        dconv = self._dilated_conv(cur, dil, wpos)
                        sig, skp = self._chan_reduce(dconv)
                        cur = tf.add(cur, sig, name='residual_add')
                        skps.append(skp)
                        aop = tf.assign(dst_buf[wpos + dil], cur)
                        with tf.control_dependencies([aop]):
                            cur_buf = dst_buf # will this work?  doesn't seem like it...

        skp_all = tf.add_n(skps, name='add_skip')
        logits, softmax_out = self._postprocess(skp_all)
        with tf.variable_scope('postprocess', reuse=None):
            wav_buf = tf.get_variable('output',
                    [self.batch_sz, self.chunk_sz],
                    initializer=tf.zeros_initializer,
                    trainable=False,
                    collections=just_loop_bufs)

        samp, hot, wav_val = self._sample_next(logits)

        wop = tf.assign(wav_buf[:,wpos], wav_val)
        with tf.control_dependencies([wop]):
            wav_nxt, wpos_nxt = tf.cond(tf.equal(wpos + 1, self.chunk_sz),
                    lambda: (self._next_window(wav_buf, wav), 0),
                    lambda: (wav, wpos + 1))

        iop = tf.assign(in_buf[:,wpos_nxt], hot)
        with tf.control_dependencies([iop]):
            return i + 1, wav_nxt, wpos_nxt

    def init_buffers(self, sess):
        '''initialize all non-trainable variable buffers used during inference'''
        sess.run(self.loop_buf_init)



    def build_graph(self):
        '''create the whole inference graph, which is capable of generating some
        number of waveforms incrementally in parallel'''
        with tf.variable_scope('preprocess', reuse=None): 
            self._non_loop_init()

        seed = tf.zeros([self.batch_sz, 0])
        i, waveform, wpos = tf.while_loop(
                self._loop_cond,
                self._loop_body,
                loop_vars=(0, seed, 0),
                shape_invariants=(
                    tf.TensorShape(()),
                    tf.TensorShape([self.batch_sz, None]),
                    tf.TensorShape(())),
                parallel_iterations=1,
                back_prop=False
                )

        loop_bufs = tf.get_collection(self.loop_buf_coll)
        self.loop_buf_init = tf.variables_initializer(loop_bufs)

        self.graph_built = True
        return i, waveform, wpos



