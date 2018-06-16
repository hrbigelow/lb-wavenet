import tensorflow as tf


def mu_encode(x, mu, n_quanta):
    '''mu-law encode and quantize'''
    mu_ten = tf.to_float(mu)
    amp = tf.sign(x) * tf.log1p(mu * tf.abs(x)) / tf.log1p(mu)
    quant = (amp + 1) * 0.5 * mu + 0.5

class WaveNet(object):

    def __init__(self,
            n_blocks,
            n_block_layers,
            n_in_chan,
            n_res_chan,
            n_dil_chan):

        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_in_chan = n_in_chan
        self.n_res_chan = n_res_chan
        self.n_dil_chan = n_dil_chan
        self.filters = []
        self.saved = []
        

    def create_filter(self, name, shape):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        variable = tf.Variable(initializer(shape=shape), name=name)
        return variable


    def preprocess(self):
        input_shape = [None, None, self.n_in_chan]
        filter_shape = [1, self.n_in_chan, self.n_res_chan]

        with tf.name_scope('inputs'):
            self.raw_input = tf.placeholder(tf.float32, input_shape, name='raw')
            self.is_first = tf.placeholder(tf.int32, [], name='is_first')

        with tf.name_scope('preprocess'):
            filt = self.create_filter('in_filter', filter_shape)
            pre_op = tf.nn.convolution(self.raw_input, filt,
                    'VALID', [1], [1], 'in_conv')

        self.filters.append(filt)
        return pre_op

    
    def dilated_conv(self, prev_op, dilation): 
        '''construct one dilated, gated convolution as in equation 2 of WaveNet Sept 2016'''
        batch_sz = tf.shape(self.raw_input)[0]
        conv_shape = [2, self.n_res_chan, self.n_dil_chan]
        saved_shape = [batch_sz, dilation, self.n_res_chan]
        prepend_len = dilation * (1 - self.is_first)
        
        with tf.name_scope('dilated_conv'):
            signal_filt = self.create_filter('filter', conv_shape) 
            gate_filt = self.create_filter('gate', conv_shape) 
            self.filters.append(signal_filt)
            self.filters.append(gate_filt)
            save = tf.Variable(tf.zeros(saved_shape), name='saved_length%i' % dilation,
                    validate_shape = False)
            self.saved.append(save)
            stop_grad = tf.stop_gradient(save)
            concat = tf.concat([stop_grad[:,0:prepend_len,:], prev_op], 
                    1, name='concat')
            assign = save.assign(concat[:,-dilation:,:])

            signal = tf.nn.convolution(concat, signal_filt, 
                    'VALID', [1], [dilation], 'signal_dilation%i' % dilation)
            post_signal = tf.tanh(signal)
            gate = tf.nn.convolution(concat, gate_filt,
                    'VALID', [1], [dilation], 'gate_dilation%i' % dilation)
            post_gate = tf.sigmoid(gate)

            with tf.control_dependencies([assign]):
                z = post_signal + post_gate
                
        return z 


    def chan_reduce(self, prev_op):
        shape = [1, self.n_dil_chan, self.n_res_chan]

        with tf.name_scope('chan_reduce'):
            filt = self.create_filter('dense_filter', shape)
            self.filters.append(filt)
            chan = tf.nn.convolution(prev_op, filt,
                    'VALID', [1], [1], 'conv')
        return chan


    def create_graph(self, graph):
        '''main wavenet structure'''

        with graph.as_default():
            cur = self.preprocess()
            skip = []

            for b in range(self.n_blocks):
                with tf.name_scope('block%i' % (b + 1)):
                    for l in range(self.n_block_layers):
                        with tf.name_scope('layer%i' % (l + 1)):
                            dil = 2**l
                            dil_conv_op = self.dilated_conv(cur, dil)
                            chan_reduce_op = self.chan_reduce(dil_conv_op)
                            skip.append(chan_reduce_op)
                            cur = cur + chan_reduce_op


            self.saved_init = tf.variables_initializer(self.saved, 'saved_init')
            self.filters_init = tf.variables_initializer(self.filters, 'filters_init')
        self.output = cur 


    def predict(self, sess, wave, beg, end, is_first):
        '''calculate top-level convolutions of a subrange of wave.
        if _is_first_run, previous state is re-initialized'''
        wave_window = wave[:,beg:end,:]
        if (is_first):
            sess.run(self.saved_init,
                    feed_dict = { self.raw_input: wave })

        ret = sess.run(self.output, 
                feed_dict = {
                    self.raw_input: wave_window,
                    self.is_first: int(is_first) 
                    })
        return ret

