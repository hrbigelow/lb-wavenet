import tensorflow as tf
from enum import IntEnum


class ArchCat(IntEnum):
    PRE = 1 
    RESIDUAL = 2 
    SKIP = 3
    SIGNAL = 4 
    GATE = 5 
    GC_SIGNAL = 6
    GC_GATE = 7 
    GC_EMBED = 8
    POST1 = 9
    POST2 = 10


class WaveNetArch(object):

    '''Provides all parameters needed to fully determine a model, either for
    training or inference.  Manages all trainable variables and their saving
    and restoring'''

    def __init__(self,
            n_blocks,
            n_block_layers,
            n_quant,
            n_res,
            n_dil,
            n_skip,
            n_post,
            n_gc_embed,
            n_gc_category):

        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_quant = n_quant
        self.n_res = n_res
        self.n_dil = n_dil
        self.n_skip = n_skip
        self.n_post = n_post
        self.n_gc_embed = n_gc_embed
        self.n_gc_category = n_gc_category
        self.use_gc = n_gc_embed > 0
        self.graph_built = False
        self.saver = None

        self.shape = {
                ArchCat.PRE: [1, self.n_quant, self.n_res],
                ArchCat.RESIDUAL: [1, self.n_dil, self.n_res],
                ArchCat.SKIP: [1, self.n_dil, self.n_skip],
                ArchCat.SIGNAL: [2, self.n_res, self.n_dil],
                ArchCat.GATE: [2, self.n_res, self.n_dil],
                ArchCat.GC_SIGNAL: [1, self.n_gc_embed, self.n_dil],
                ArchCat.GC_GATE: [1, self.n_gc_embed, self.n_dil],
                ArchCat.GC_EMBED: [self.n_gc_category, self.n_gc_embed],
                ArchCat.POST1: [1, self.n_skip, self.n_post],
                ArchCat.POST2: [1, self.n_post, self.n_quant]
                }

    def get_variable(self, arch):
        '''wrapper for tf.get_variable that supplies the proper shape and name
        according to arch'''
        return tf.get_variable(arch.name, self.shape[arch])

    def _arch_string(self):
        '''generate a string corresponding to the architecture
        parameters of the model'''
        return 'b{}_l{}_q{}_r{}_d{}_s{}_p{}_e{}_c{}'.format(
                self.n_blocks,
                self.n_block_layers,
                self.n_quant,
                self.n_res,
                self.n_dil,
                self.n_skip,
                self.n_post,
                self.n_gc_embed,
                self.n_gc_category)

    def _maybe_init_saver(self):
        if not self.graph_built:
            raise ValueError
        if self.saver is None:
            self.saver = tf.train.Saver(
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def save(self, sess, logdir, file_pfx, step):
        '''saves trainable variables, generating a special filename
        that encodes architectural parameters'''
        self._maybe_init_saver()
        fn = '{}/{}.{}'.format(logdir, file_pfx, self._arch_string())
        path_pfx = self.saver.save(fn, sess, step)
        return path_pfx


    def restore(self, sess, logdir, file_pfx, step):
        '''finds the appropriate checkpoint file saved by 'save' and loads into
        the existing graph'''
        self._maybe_init_saver()
        fn = '{}/{}.{}'.format(logdir, file_pfx, self._arch_string())
        self.saver.restore(sess, fn)
    


