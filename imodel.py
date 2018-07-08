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


