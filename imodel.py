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

    def _preprocess(self, prev_val):
        with tf.name_scope('preprocess'):
            if self.use_gc:
                gc_tab = self.get_var('GE', ar.ArchCat.PRE)
            filt = self.get_var('QR', ar.ArchCat.PRE)
            pre_op = tf.matmul(prev_val, filt[0]) + tf.matmul(prev_val, filt[1])
        return pre_op

    def _dilated_conv(self, prev_op, dilation, pos, block, layer):
        '''pos: zero-based position in the cached windoow'''
        v = {}
        gc = {}
        cache = self.get_cache(block, layer)
        embed = self.get_embed()
        for arch = [ar.ArchCat.SIGNAL, ar.ArchCat.GATE]:
            filt = self.get_var('RD', arch, block, layer)
            with tf.name_scope(arch.name):
                v[arch] = tf.matmul(cache[pos], filt[0]) + tf.matmul(prev_op, filt[1])
                if self.use_gc:
                    gc_filt = self.get_var('ED', arch, block, layer)
                    gc[arch] = tf.matmul(embed, gc_filt[0]) + tf.matmul(embed, gc_filt[1])
           
        (signal, gate) = (v[ar.ArchCat.SIGNAL], v[ar.ArchCat.GATE])
        if self.use_gc:
            signal = tf.add(signal, gc[ar.ArchCat.SIGNAL], 'add_gc_signal')
            gate = tf.add(gate, gc[ar.ArchCat.GATE], 'add_gc_gate')

        post_signal = tf.tanh(signal)
        post_gate = tf.sigmoid(gate)

        z = post_signal * post_gate

        return z

    def _chan_reduce(self, prev_op, block, layer):
        sig_filt = self.get_var('DR', ar.ArchCat.RESIDUAL, block, layer)
        skip_filt = self.get_var('DS', ar.ArchCat.SKIP, block, layer)
        next_cache = self.get_next_cache(block, layer)
        with tf.name_scope('signal'):
            signal = tf.matmul(prev_op, sig_filt[0]) + tf.matmul(prev_op, sig_filt[1])
        with tf.name_scope('skip'):
            skip = tf.matmul(prev_op, skip_filt[0]) + tf.matmul(prev_op, skip_filt[1])
        return signal, skip 


