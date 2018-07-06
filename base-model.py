import tensorflow as tf

def create_var(shape, name=None):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


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

        self.vars['QR'] = create_var(shape['QR'])
        if self.use_gc:
            self.vars['GE'] = create_var(shape['GE'])

        def _make_shape(shape):
            return [[create_var(shape)
            for _ in range(self.n_block_layers)]
            for _ in range(self.n_blocks)]

        keys = ['sRD', 'gRD', 'DR', 'DS']
        if self.use_gc:
            keys += ['sED', 'gED']

        for key in keys 
            self.vars[key] = _make_shape(shape[key])

        self.vars['SP'] = create_var(shape['SP'])
        self.vars['PQ'] = create_var(shape['PQ'])



    def get_var(self, part, block=0, layer=0):
        '''retrieve a variable from the model.
        Args:
        part: one of: QR, GE, sRD, gRD, sED, gED, DR, DS, SP, PQ
        block: integer (>= 0)
        layer: integer (>= 0)
        block and layer are ignored for QR, GE, SP and PQ
        '''
        item = self.vars[part]
        if isinstance(item, list):
            return item[block][layer]
        else:
            return item





