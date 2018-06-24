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
            n_dil_chan,
            n_skip_chan,
            n_post1_chan,
            n_quant_chan):

        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_in_chan = n_in_chan
        self.n_res_chan = n_res_chan
        self.n_dil_chan = n_dil_chan
        self.n_skip_chan = n_skip_chan
        self.n_post1_chan = n_post1_chan
        self.n_quant_chan = n_quant_chan
        self.filters = []
        self.saved = []
        

    def create_filter(self, name, shape):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        variable = tf.Variable(initializer(shape=shape), name=name)
        return variable


    def preprocess(self):
        input_shape = [None, None, self.n_in_chan]
        filter_shape = [1, self.n_in_chan, self.n_res_chan]

        with tf.name_scope('data_input'):
            self.raw_input = tf.placeholder(tf.float32, input_shape, name='raw')

        with tf.name_scope('config_inputs'):
            self.batch_sz = tf.shape(self.raw_input)[0]

        with tf.name_scope('preprocess'):
            filt = self.create_filter('in_filter', filter_shape)
            pre_op = tf.nn.convolution(self.raw_input, filt,
                    'VALID', [1], [1], 'in_conv')

        self.filters.append(filt)
        return pre_op

    
    def dilated_conv(self, prev_op, dilation, batch_sz): 
        '''construct one dilated, gated convolution as in equation 2 of WaveNet Sept 2016'''
    
        # save last postiions from prev_op
        with tf.name_scope('load_store_values'):
            saved_shape = [batch_sz, dilation, self.n_res_chan]
            save = tf.Variable(tf.zeros(saved_shape), name='saved_length%i' % dilation,
                    validate_shape = False)
            stop_grad = tf.stop_gradient(save)
            concat = tf.concat([stop_grad, prev_op], 1, name='concat')
            assign = save.assign(concat[:,-dilation:,:])
        self.saved.append(save)

        # construct signal and gate logic
        conv_shape = [2, self.n_res_chan, self.n_dil_chan]
        with tf.name_scope('signal'):
            signal_filt = self.create_filter('filter', conv_shape) 
            signal = tf.nn.convolution(concat, signal_filt, 
                    'VALID', [1], [dilation], 'dilation%i_conv' % dilation)

        with tf.name_scope('gate'):
            gate_filt = self.create_filter('filter', conv_shape) 
            gate = tf.nn.convolution(concat, gate_filt,
                    'VALID', [1], [dilation], 'dilation%i_conv' % dilation)

        self.filters.append(signal_filt)
        self.filters.append(gate_filt)

        post_signal = tf.tanh(signal)
        post_gate = tf.sigmoid(gate)

        with tf.control_dependencies([assign]):
            z = post_signal * post_gate
                
        return z 


    def chan_reduce(self, prev_op):
        sig_shape = [1, self.n_dil_chan, self.n_res_chan]
        skip_shape = [1, self.n_dil_chan, self.n_skip_chan]
        with tf.name_scope('signal_%i_to_%i' % (sig_shape[1], sig_shape[2])):
            sig_filt = self.create_filter('filter', sig_shape)
            self.filters.append(sig_filt)
            signal = tf.nn.convolution(prev_op, sig_filt,
                    'VALID', [1], [1], 'conv')

        with tf.name_scope('skip_%i_to_%i' % (skip_shape[1], skip_shape[2])):
            skip_filt = self.create_filter('filter', skip_shape)
            self.filters.append(skip_filt)
            skip = tf.nn.convolution(prev_op, skip_filt,
                    'VALID', [1], [1], 'conv')

        return signal, skip 

    def postprocess(self, prev_op):
        '''implement the post-processing, just after the '+' sign and
        before the 'ReLU', where all skip connections add together.
        see section 2.4'''
        shape1 = [1, self.n_skip_chan, self.n_post1_chan]
        shape2 = [1, self.n_post1_chan, self.n_quant_chan]
        with tf.name_scope('postprocess'):
            relu1 = tf.nn.relu(prev_op, 'ReLU')
            with tf.name_scope('chan_%i_to_%i' % (shape1[1], shape1[2])):
                filt1 = self.create_filter('filter', shape1)
                dense1 = tf.nn.convolution(relu1, filt1, 'VALID', [1], [1], 'conv')

            relu2 = tf.nn.relu(dense1, 'ReLU')
            with tf.name_scope('chan_%i_to_%i' % (shape2[1], shape2[2])):
                filt2 = self.create_filter('filter', shape2)
                dense2 = tf.nn.convolution(relu2, filt2, 'VALID', [1], [1], 'conv')

            with tf.name_scope('softmax'):
                softmax = tf.nn.softmax(dense2, 0, 'softmax')

        self.filters.append(filt1)
        self.filters.append(filt2)
        return softmax


    def create_graph(self):
        '''main wavenet structure'''

        graph = tf.Graph()
        with graph.as_default():
            cur = self.preprocess()
            skip = []

            for b in range(self.n_blocks):
                with tf.name_scope('block%i' % (b + 1)):
                    for l in range(self.n_block_layers):
                        with tf.name_scope('layer%i' % (l + 1)):
                            dil = 2**l
                            dil_conv_op = self.dilated_conv(cur, dil, self.batch_sz)
                            (signal_op, skip_op) = self.chan_reduce(dil_conv_op)
                            skip.append(skip_op)
                            new_win = tf.shape(signal_op)[1]
                            cur = tf.add(cur, signal_op, name='residual_add') 
            sum_all = sum(skip)
            out = self.postprocess(sum_all)
            self.saved_init = tf.variables_initializer(self.saved, 'saved_init')
            self.filters_init = tf.variables_initializer(self.filters, 'filters_init')
        self.output = out 
        return graph


    def predict(self, sess, wave, beg, end):
        '''calculate top-level convolutions of a subrange of wave.
        if _is_first_run, previous state is re-initialized'''
        wave_window = wave[:,beg:end,:]

        ret = sess.run(self.output, 
                feed_dict = {
                    self.raw_input: wave_window
                    })
        return ret

