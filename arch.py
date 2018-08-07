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
            n_gc_category,
            use_bias,
            add_summary):

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
        self.add_summary = add_summary
        self.use_bias = use_bias
        self.filter_init = tf.contrib.layers.xavier_initializer_conv2d()
        self.bias_init = tf.constant_initializer(value=0.0, dtype=tf.float32)

        # dict for use with save/restore 
        self.vars = {}

        self.shape = {
                ArchCat.PRE: [self.n_quant, self.n_res],
                ArchCat.RESIDUAL: [self.n_dil, self.n_res],
                ArchCat.SKIP: [self.n_dil, self.n_skip],
                ArchCat.SIGNAL: [2, self.n_res, self.n_dil],
                ArchCat.GATE: [2, self.n_res, self.n_dil],
                ArchCat.GC_SIGNAL: [self.n_gc_embed, self.n_dil],
                ArchCat.GC_GATE: [self.n_gc_embed, self.n_dil],
                ArchCat.GC_EMBED: [self.n_gc_category, self.n_gc_embed],
                ArchCat.POST1: [self.n_skip, self.n_post],
                ArchCat.POST2: [self.n_post, self.n_quant]
                }

    def get_variable(self, arch, *var_indices, get_bias=False):
        '''wrapper for tf.get_variable that associates arch name with a shape
        
        arch:     enumeration specifying shape and semantics
        get_bias: if True, retrieve the bias variable corresponding to this variable
        *args:    zero or more integers that denote distinct instances of the variable,
                  used for architectures with repetitive structure
        '''
        if get_bias:
            name = arch.name + '_BIAS'
            shape = self.shape[arch][-1]
            init = self.bias_init
        else:
            name = arch.name
            shape = self.shape[arch]
            init = self.filter_init

        var = tf.get_variable(name, shape, initializer=init)
        serial_name = '_'.join(map(str, [name, *var_indices]))

        # make sure that serial_name and var are both new or both not new
        sn_exists = serial_name in self.vars
        var_exists = var in self.vars.values()
        if sn_exists != var_exists:
            from sys import stderr
            if sn_exists:
                print('Attempting to store {} under {}.\n' 
                        'Variable {} already stored there.'.format(
                            var.op.name, serial_name, self.vars[serial_name].op.name),
                        file=stderr)
                exit(1)
            if var_exists:
                existing_sn = next((v for k, v in self.vars.items() if v == var), None)
                print('Attempting to store {} under {}.\n' 
                        'Already stored under {}'.format(
                            var.name, serial_name, existing_sn),
                        file=stderr)
                exit(1)
        
        self.vars[serial_name] = var

        if self.add_summary:
            tf.summary.histogram(name, var)
        return var

    def _maybe_init_saver(self):
        if not self.graph_built:
            raise ValueError
        if self.saver is None:
            self.saver = tf.train.Saver(self.vars)

    def _json_stem(self, json_file):
        import re
        m = re.fullmatch('(.+)\.json', json_file)
        return m.group(1)

    def save(self, sess, arch_pfx, step):
        '''saves trainable variables, generating a special filename
        that encodes architectural parameters'''
        self._maybe_init_saver()
        path_pfx = self.saver.save(sess, arch_pfx, step)
        return path_pfx

    @staticmethod
    def expand_ckpt(ckpt):
        '''expand ckpt into list of ckpt files that the writer would generate'''
        suffixes = ['index', 'meta', 'data-00000-of-00001']
        return ['{}.{}'.format(ckpt, s) for s in suffixes]


    def restore(self, sess, ckpt_file):
        '''finds the appropriate checkpoint file saved by 'save' and loads into
        the existing graph'''
        from sys import stderr
        from os import access, R_OK
        for fn in self.expand_ckpt(ckpt_file):
            if not access(fn, R_OK):
                print("Couldn't find checkpoint file {}".format(fn), file=stderr)
                exit(1)
        self._maybe_init_saver()
        self.saver.restore(sess, ckpt_file)
    


