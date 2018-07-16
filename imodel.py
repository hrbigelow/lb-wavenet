import tensorflow as tf
import arch as ar

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
            l2_factor,
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
                n_gc_category)

        self.chunk_sz = chunk_sz
        # create all lookback buffers
        self.input = tf.Variable(tf.zeros([None, self.chunk_sz + 1]),
                name='lb_in',
                trainable=False)
        self.lookback = [tf.Variable(tf.zeros([None, 2**l + self.chunk_sz]),
            name='lb_b{}_l{}'.format(b, l),
            trainable=False)
            for b in range(self.n_blocks)
            for l in range(self.n_block_layers)]


    def _preprocess(self, prev_val):
        '''everything done to the raw signal before it is
        ready for the layers.'''  
        with tf.name_scope('preprocess'):
            if self.use_gc:
                gc_tab = self.get_var('GE', ar.ArchCat.PRE)
            filt = self.get_var('QR', ar.ArchCat.PRE)
            pre_op = tf.matmul(prev_val, filt[0]) + tf.matmul(prev_val, filt[1])
        return pre_op


    def _dilated_conv(self, src_buf, prev_val, pos):
        '''
        pos: zero-based position in the cached window
        prev_val: value from previous layer at time t 
        '''
        embed = self.get_embed()

        v = {}
        for arch in [ar.ArchCat.SIGNAL, ar.ArchCat.GATE]:
            filt = self.get_variable(arch)
            v[arch] = tf.matmul(src_buf[pos], filt[0]) \
            + tf.matmul(prev_val, filt[1])

        if self.use_gc:
            for arch in [ar.ArchCat.GC_SIGNAL, ar.ArchCat.GC_GATE]: 
                gc_filt = self.get_variable(arch)
                gc_proj = tf.matmul(embed, gc_filt[0]) + tf.matmul(embed, gc_filt[1])
                v[arch] = tf.add(v[arch], gc_proj, 'add')

        z = tf.tanh(v[ar.ArchCat.SIGNAL]) * tf.sigmoid(v[ar.ArchCat.GATE])
        return z


    def _chan_reduce(self, prev_op):
        '''simply provide the channel reducing operations'''
        sig_filt = self.get_variable(ar.ArchCat.RESIDUAL)
        skp_filt = self.get_variable(ar.ArchCat.SKIP)

        with tf.name_scope('signal'):
            signal = tf.matmul(prev_op, sig_filt)
        with tf.name_scope('skip'):
            skip = tf.matmul(prev_op, skip_filt)
        return signal, skip 

    def _postprocess(self, prev_op):
        '''implement post-processing, just after the '+' sign and
        before the 'ReLU'.  See section 2.4'''
        post1_filt = self.get_variable(ar.ArchCat.POST1)
        post2_filt = self.get_variable(ar.ArchCat.POST2)

        with tf.name_scope('postprocess'):
            relu1 = tf.nn.relu(prev_op, 'ReLU')
            with tf.name_scope('chan'):
                dense1 = tf.matmul(relu1, post1_filt)

            relu2 = tf.nn.relu(dense1, 'ReLU')
            with tf.name_scope('chan'):
                dense2 = tf.matmul(relu2, post2_filt)

            with tf.name_scope('softmax'):
                softmax = tf.nn.softmax(dense2, 0, 'softmax')

        return dense2, softmax

    def _sample_next(self, logits_op, wpos):
        '''obtain a random sample from the most recent softmax output,
        one-hot encode it, and populate the next input element
        '''
        samp = tf.multinomial(logits_op, 1, self.rand_seed)
        hot = tf.one_hot(samp, self.n_quant)
        in_buf = tf.get_variable('input', reuse=True)
        aop = tf.assign(in_buf[wpos + 1], hot)
        with tf.control_dependencies([aop]):
            pass            



    def _next_window(self, out):
        '''called after each chunk_sz iterations of _loop_body.
        appends the populated window to the final output buffer,
        and advances all lookback buffers.
        '''
        lb_vars = [v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                if 'lookback' in v.name]
        # copy len-chunk_sz last values to beginning
        aops = []
        for v in lb_vars:
            op = tf.assign(v, v[:,self.batch_sz:,:])
            aops.append(op)

        out_buf = tf.get_variable('output', reuse=True)
        with tf.control_dependencies(aops):
            out = tf.concat([out, out_buf]) 
        return out


    def _loop_cond(self, out, wpos, i):
        return i < self.gen_sz 


    def _loop_body(self, out, wpos, i):
        '''while loop body for inference'''
        in_buf = tf.get_variable('input', [None, self.chunk_sz, self.n_quant],
            reuse=None)
        in2_buf = self._preprocess(in_buf)
        cur_buf = in2_buf

        skps = []
        for b in range(self.n_blocks):
            for bl in range(self.n_block_layers):
                l = b * self.n_block_layers + bl
                dil = 2**bl
                with tf.variable_scope('dconv{}'.format(l), resuse=None):
                    dst_buf = tf.get_variable('lookback',
                            [None, dil + self.chunk_sz, self.n_res])
                    dconv = self._dilated_conv(cur_buf, cur, wpos)
                    sig, skp = self._chan_reduce(dconv)
                    cur = tf.add(cur, sig, name='residual_add')
                    skps.append(skp)
                    aop = tf.assign(dst_buf[wpos + dil], cur)
                    with tf.control_dependencies([aop]):
                        cur_buf = dst_buf # will this work?  doesn't seem like it...

        skp_all = sum(skps)
        logits, softmax_out = self._postprocess(skp_all)
        out_buf = tf.get_variable('output',
                [None, self.chunk_sz, self.n_quant])
        aop = tf.assign(out_buf[wpos], softmax_out)
        with tf.control_dependencies([aop]):
            self._sample_next(logits, wpos)

        wpos = wpos + 1
        if wpos == self.chunk_sz:
            out = self._next_window(out)
            wpos = 0

        return out, wpos, i 


    def create_inference_graph(self):
        '''create the whole inference graph, which is capable of generating some
        number of waveforms incrementally in parallel'''
        waveform, _, _ = tf.while_loop(self._loop_cond, self._loop_body, ([], 0, 0))
        return waveform



