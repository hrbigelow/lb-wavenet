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


    def _dilated_conv(self, src_buf, dilation, pos):
        '''pos: zero-based position in the cached windoow'''
        embed = self.get_embed()

        v = {}
        for arch in [ar.ArchCat.SIGNAL, ar.ArchCat.GATE]:
                filt = self.get_variable(arch)
                v[arch] = tf.matmul(src_buf[pos], filt[0])
                + tf.matmul(src_buf[pos + dilation], filt[1])

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

    def _loop_cond(self, wpos, cont):
        return cont 


    def _loop_body(self, wpos, cont):
        for b in range(self.n_blocks):
            for bl in range(self.n_block_layers):
                l = b * self.n_block_layers + bl
                dil = 2**bl
                with tf.variable_scope('dconv{}'.format(l)):
                    dconv = self._dilated_conv(cur_buf, dil, wpos)

        dst_buf = tf.get_variable('lookback_{}'.format(layer),
                [None, dilation + self.chunk_sz, self.n_res])
        scope = 'dconv{}'.format(layer)
        op = tf.assign(dst_buf[pos], z)
        with tf.control_dependencies([op]):
            pass
        with tf.variable_scope(scope, reuse=None):
            pass

