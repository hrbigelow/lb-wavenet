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
                n_gc_category)

        self.chunk_sz = chunk_sz
        self.batch_sz = batch_sz

    def _non_loop_init(self):
        '''operations that will run before the loop starts'''
        if self.use_gc:
            self.gc_ids = tf.placeholder(tf.int32, (self.batch_sz,), 'gc_ids')
            gc_tab = self.get_variable(ar.ArchCat.GC_EMBED)
            self.gc_embeds = tf.gather(gc_tab, self.gc_ids)
            
        self.gen_sz = tf.placeholder(tf.int32, (), 'gen_sz')


    def _preprocess(self, src_buf, gpos):
        '''everything done to the raw signal before it is
        ready for the layers.'''  
        filt = self.get_variable(ar.ArchCat.PRE)
        pre_op = tf.matmul(src_buf[:,gpos,:], filt[0])

        return pre_op


    def _dilated_conv(self, src_buf, prev_val, pos):
        '''
        pos: zero-based position in the cached window
        prev_val: value from previous layer at time t 
        '''
        v = {}
        sig_gate = [ar.ArchCat.SIGNAL, ar.ArchCat.GATE]
        sig_gate_gc = [ar.ArchCat.GC_SIGNAL, ar.ArchCat.GC_GATE]

        for arch in sig_gate:
            filt = self.get_variable(arch)
            v[arch] = tf.matmul(src_buf[:,pos,:], filt[0]) \
            + tf.matmul(prev_val, filt[1])

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


    def _sample_next(self, logits_op, wpos, in_buf):
        '''obtain a random sample from the most recent softmax output,
        one-hot encode it, and populate the next input element
        '''
        samp = tf.multinomial(logits_op, 1)
        hot = tf.one_hot(samp, self.n_quant)
        aop = tf.assign(in_buf[wpos + 1], hot)
        return aop


    def _next_window(self, out_buf, out):
        '''called after each chunk_sz iterations of _loop_body.
        appends the populated window to the final output buffer,
        and advances all lookback buffers.
        returns: tensor of shape [None, total_length]
        '''
        lb_vars = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                if 'lookback' in v.name]
        # copy len-chunk_sz last values to beginning
        aops = []
        for v in lb_vars:
            op = tf.assign(v, v[:,self.chunk_sz:,:])
            aops.append(op)

        with tf.control_dependencies(aops):
            out = tf.concat([out, out_buf]) 
        return out


    def _loop_cond(self, i, out, wpos):
        return i < self.gen_sz 


    def _loop_body(self, i, out, wpos):
        '''while loop body for inference'''
        with tf.variable_scope('preprocess', reuse=None):
            in_buf = tf.get_variable('input',
                    [self.batch_sz, self.chunk_sz, self.n_quant],
                    trainable=False)
            pre_buf = tf.get_variable('input_trans',
                    [self.batch_sz, self.chunk_sz, self.n_res],
                    trainable=False)
            cur = self._preprocess(in_buf, wpos)

        aop = tf.assign(pre_buf[wpos], cur)
        with tf.control_dependencies([aop]):
            cur_buf = pre_buf 

        skps = []
        for b in range(self.n_blocks):
            for bl in range(self.n_block_layers):
                l = b * self.n_block_layers + bl
                dil = 2**bl
                with tf.variable_scope('dconv{}'.format(l), reuse=None):
                    dst_buf = tf.get_variable('lookback',
                            [self.batch_sz, dil + self.chunk_sz, self.n_res],
                            trainable=False)
                    dconv = self._dilated_conv(cur_buf, cur, wpos)
                    sig, skp = self._chan_reduce(dconv)
                    cur = tf.add(cur, sig, name='residual_add')
                    skps.append(skp)
                    aop = tf.assign(dst_buf[wpos + dil], cur)
                    with tf.control_dependencies([aop]):
                        cur_buf = dst_buf # will this work?  doesn't seem like it...

        skp_all = sum(skps)
        logits, softmax_out = self._postprocess(skp_all)
        with tf.variable_scope('postprocess', reuse=None):
            out_buf = tf.get_variable('output',
                    [self.batch_sz, self.chunk_sz, self.n_quant],
                    trainable=False)
        aop = tf.assign(out_buf[wpos], softmax_out)
        with tf.control_dependencies([aop]):
            sop = self._sample_next(logits, wpos, in_buf)

        with tf.control_dependencies([sop]):
            wpos = wpos + 1

        wpos = tf.Print(wpos, [wpos], 'Wpos=')
        if wpos == self.chunk_sz:
            out = self._next_window(out_buf, out)
            wpos = 0

        return i + 1, out, wpos


    def build_graph(self):
        '''create the whole inference graph, which is capable of generating some
        number of waveforms incrementally in parallel'''
        with tf.variable_scope('preprocess', reuse=None): 
            self._non_loop_init()

        _, waveform, _ = tf.while_loop(self._loop_cond, self._loop_body,
                (0, [], 0))
        self.graph_built = True
        return waveform



