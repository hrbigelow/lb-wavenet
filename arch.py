import tensorflow as tf
from enum import IntEnum
import ckpt


class ArchCat(IntEnum):
    PRE = 1 
    LC_UPSAMPLE = 2
    RESIDUAL = 3 
    SKIP = 4
    SIGNAL = 5 
    GATE = 6 
    GC_SIGNAL = 7
    GC_GATE = 8 
    GC_EMBED = 9
    LC_SIGNAL = 10 
    LC_GATE = 11
    POST1 = 12 
    POST2 = 13
    SAVE = 14
    GLOBAL_STEP = 15 # global training step count
    VALID_SAMPLES = 16 # number of valid samples seen so far


class WaveNetArch(ckpt.Checkpoint):

    '''Provides all parameters needed to fully determine a model, either for
    training or inference.  Manages all trainable variables and their saving
    and restoring'''

    def __init__(self,
            batch_sz,
            n_quant,
            n_res,
            n_dil,
            n_skip,
            n_post,
            n_gc_embed,
            n_gc_category,
            n_lc_in,
            n_lc_out,
            add_summary,
            # checkpoint related
            n_keep_checkpoints,
            ckpt_path,
            resume_step,
            sess=None 
            ):
        super().__init__(ckpt_path, n_keep_checkpoints, resume_step, sess)
        self.batch_sz = batch_sz
        self.n_quant = n_quant
        self.n_res = n_res
        self.n_dil = n_dil
        self.n_skip = n_skip
        self.n_post = n_post
        self.n_gc_embed = n_gc_embed
        self.n_gc_category = n_gc_category
        self.n_lc_in = n_lc_in
        self.n_lc_out = n_lc_out
        self.add_summary = add_summary
        self.sess = sess

        self.graph_built = False
        self.vars_initialized = False
        self.saver = None
        self.filter_init = tf.contrib.layers.xavier_initializer_conv2d()
        self.bias_init = tf.constant_initializer(value=0.0, dtype=tf.float32)

        if tf.executing_eagerly():
            import tensorflow.contrib.eager as tfe
            self.container = tfe.EagerVariableStore() 


        # dict for use with save/restore 
        # contains all of the model's trainable variables
        self.vars = {}

        def _upsample_shape(i):
            if i == 0:
                dim2 = self.n_lc_in
            else:
                dim2 = self.n_lc_out 
            return [self.lc_upsample[i], self.n_lc_out, dim2]

        def _save_var_shape(dilation, *ignored):
            return [self.batch_sz, dilation, self.n_res]

        self.shape = {
                # shape, or function accepting var_indices and returning shape
                ArchCat.PRE: [self.n_quant, self.n_res],
                ArchCat.LC_UPSAMPLE: _upsample_shape, 
                ArchCat.RESIDUAL: [self.n_dil, self.n_res],
                ArchCat.SKIP: [self.n_dil, self.n_skip],
                ArchCat.SIGNAL: [2, self.n_res, self.n_dil],
                ArchCat.GATE: [2, self.n_res, self.n_dil],
                ArchCat.GC_SIGNAL: [self.n_gc_embed, self.n_dil],
                ArchCat.GC_GATE: [self.n_gc_embed, self.n_dil],
                ArchCat.GC_EMBED: [self.n_gc_category + 1, self.n_gc_embed],
                ArchCat.LC_SIGNAL: [self.n_lc_out, self.n_dil],
                ArchCat.LC_GATE: [self.n_lc_out, self.n_dil],
                ArchCat.POST1: [self.n_skip, self.n_post],
                ArchCat.POST2: [self.n_post, self.n_quant],
                ArchCat.SAVE: _save_var_shape,
                ArchCat.GLOBAL_STEP: [],
                ArchCat.VALID_SAMPLES: []
                }

    def has_global_cond(self):
        return self.n_gc_embed > 0

    def use_lc_input(self):
        return self.n_lc_out > 0


    def get_variable(self, arch, *var_indices, get_bias=False, **var_opts):
        '''wrapper for tf.get_variable that associates arch name with a shape
        
        arch:     enumeration specifying shape and semantics
        get_bias: if True, retrieve the bias variable corresponding to this variable
        *args:    zero or more integers that denote distinct instances of the variable,
                  used for architectures with repetitive structure
        '''
        if isinstance(self.shape[arch], list):
            shape = self.shape[arch]
        else:
            shape = self.shape[arch](*var_indices)

        if get_bias:
            name = arch.name + '_BIAS'
            shape = shape[-1] 
            init = self.bias_init
        else:
            name = arch.name
            init = self.filter_init

        # override default initializer
        var_opts.setdefault('initializer', init)

        if tf.executing_eagerly():
            with self.container.as_default():
                var = tf.get_variable(name, shape, **var_opts)
        else:
            var = tf.get_variable(name, shape, **var_opts)

        serial_name = '_'.join(map(str, [name, *var_indices]))

        # ensure we don't store the same variable under two different serialized names
        sn_exists = serial_name in self.vars
        var_exists = var in self.vars.values()
        if sn_exists and not var_exists:
            from sys import stderr
            print('Attempting to store variable {} under serial name {}.\n' 
                    'Variable {} already stored there.'.format(
                        var.name, serial_name, self.vars[serial_name].name),
                    file=stderr)
            exit(1)
        if var_exists and not sn_exists:
            from sys import stderr
            existing_sn = next((v for k, v in self.vars.items() if v == var), None)
            print('Attempting to store variable {} under serial name {}.\n' 
                    'Already stored under serial name {}'.format(
                        var.name, serial_name, existing_sn),
                    file=stderr)
            exit(1)
        
        self.vars[serial_name] = var

        if self.add_summary:
            tf.summary.histogram(name, var)
        return var

    def _json_stem(self, json_file):
        import re
        m = re.fullmatch('(.+)\.json', json_file)
        return m.group(1)

    def init_vars(self):
        if tf.executing_eagerly():
            pass # variables are already initialized
        else:
            self.sess.run(tf.global_variables_initializer())
            #self.sess.run([v.initializer for v in self.vars.values() if v.trainable])
        self.vars_initialized = True

