import tensorflow as tf
from enum import Enum



class ArchCat(Enum):
    SIGNAL = 's'
    GATE = 'g'
    SKIP = 'k'
    RESIDUAL = 'r'
    PRE = 'x'
    POST = 'y'


def _make_key(channel_code, arch_cat, block=None, layer=None):
    if block is None and layer is None:
        return '{}{}'.format(arch_cat.value, channel_code)
    else:
        return '{}{}_{}_{}'.format(arch_cat.value, channel_code, block, layer)


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
        self.vars = {} 
        self.var_init_func = tf.contrib.layers.xavier_initializer_conv2d()


    def create_vars(self):
        '''create all trainable model parameters.
        All variables except for the global conditioning ones are matrices
        with dimensions corresponding to a number of channels of a certain type.
        The following letter codes correspond with channels:

            Q: quantization channels
            R: residual channels
            D: dilation channels
            S: skip channels
            P: post channels
            G: global condition categories
            E: global condition embeddings

        This class stores references to all variables created in the graph under self.vars.
        The variables are given default (meaningless) names that are not scoped in the tf.Graph.
        Since this class handles the saving and restoring of variables, and graphs built from
        these variables access them through self.vars, there is no need for a second naming
        scheme.  When viewing the resulting graph in TensorBoard, it is best to detach them
        from the main graph.
            
            '''
        shape = {
                'QR': [1, self.n_quant, self.n_res],
                'DR': [1, self.n_dil, self.n_res],
                'DS': [1, self.n_dil, self.n_skip],
                'RD': [2, self.n_res, self.n_dil],
                'SP': [1, self.n_skip, self.n_post],
                'PQ': [1, self.n_post, self.n_quant],
                'GE': [self.n_gc_category, self.n_gc_embed],
                'ED': [1, self.n_gc_embed, self.n_dil]
                }


        def _make_var(channel_code, arch_cat, block=None, layer=None):
            var = tf.Variable(self.var_init_func(shape=shape[channel_code]))
            self.vars[_make_key(channel_code, arch_cat, block, layer)] = var

        _make_var('QR', ArchCat.PRE)
        _make_var('SP', ArchCat.POST)
        _make_var('PQ', ArchCat.POST)

        if self.use_gc:
            _make_var('GE', ArchCat.PRE)
        
        # create all layer-based variables
        def _make_var_grid(channel_code, arch_cat):
            for b in range(self.n_blocks):
                for l in range(self.n_block_layers):
                    _make_var(channel_code, arch_cat, b, l)

        codes = ['RD', 'RD', 'DR', 'DS']
        archs = [ArchCat.SIGNAL, ArchCat.GATE, ArchCat.RESIDUAL, ArchCat.SKIP]
        if self.use_gc:
            codes += ['ED', 'ED']
            archs += [ArchCat.SIGNAL, ArchCat.GATE]

        for (code, arch) in zip(codes, archs): 
            _make_var_grid(code, arch)

        # Create the initializer
        self.var_init_op = tf.variables_initializer(list(self.vars.values()))
        

    def get_var(self, channel_code, arch_cat, block=None, layer=None):
        '''retrieve a variable from the model.
        Args:
        channel_code: QR, GE, RD, ED, DR, DS, SP, PQ
        arch_cat: ArchCat 
        block and layer are ignored for QR, GE, SP and PQ
        '''
        key = _make_key(channel_code, arch_cat, block, layer)
        item = self.vars[key]
        return item






