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
            batch_sz,
            n_in_chan,
            n_res_chan,
            n_dil_chan):

        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_in_chan = n_in_chan
        self.n_res_chan = n_res_chan
        self.n_dil_chan = n_dil_chan
        self.batch_sz = batch_sz
        

    def create_filter(self, name, shape):
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        variable = tf.Variable(initializer(shape=shape), name=name)
        return variable
        

    def create_graph(self, graph):
        '''main wavenet structure'''

        with graph.as_default():
            in_shape = [self.batch_sz, None, self.n_in_chan]
            self.raw_input = tf.placeholder(tf.float32, in_shape)

            all_filters = []

            in_filter_shape = [1, self.n_in_chan, self.n_res_chan]
            with tf.name_scope('channel_convert'):
                in_filter = self.create_filter('in_filter', in_filter_shape)
                cur = tf.nn.convolution(self.raw_input, in_filter, 'VALID', [1], [1], 'in_conv')

            all_filters.append(in_filter)

            # filters[b][l] = [pos][res_chan][dil_chan]
            filter_shape = [2, self.n_res_chan, self.n_dil_chan]
            filters = [[None] * self.n_block_layers] * self.n_blocks

            # dense_filters[b][l] = [0][out_chan][in_chan]
            dense_filter_shape = [1, self.n_dil_chan, self.n_res_chan]
            dense_filters = [[None] * self.n_block_layers] * self.n_blocks
            
            # saved[b][l] = [batch][pos][out_chan]
            saved = [[None] * self.n_block_layers] * self.n_blocks
            self.all_saved = []

            for b in range(self.n_blocks):
                with tf.name_scope('block%i' % b):
                    for l in range(self.n_block_layers):
                        with tf.name_scope('layer%i' % l):
                            dil = 2**l
                            saved_shape = [self.batch_sz, dil, self.n_res_chan]

                            with tf.name_scope('dilated_conv'):
                                filters[b][l] = self.create_filter('filter', filter_shape) 
                                saved[b][l] = tf.Variable(tf.zeros(saved_shape), name='saved')
                                concat = tf.concat([tf.stop_gradient(saved[b][l]), cur], 1,
                                        name='concat')
                                assign = saved[b][l].assign(concat[:,-dil:,:])
                                with tf.control_dependencies([assign]):
                                    dil_conv = tf.nn.convolution(concat, filters[b][l], 
                                            'VALID', [1], [dil], 'conv')

                            with tf.name_scope('chan_reduce'):
                                dense_filters[b][l] = self.create_filter('dense_filter', 
                                        dense_filter_shape)
                                chan = tf.nn.convolution(dil_conv, dense_filters[b][l],
                                        'VALID', [1], [1], 'conv')

                            self.all_saved.append(saved[b][l])
                            all_filters.append(filters[b][l])
                            all_filters.append(dense_filters[b][l])
                            cur = chan

            self.saved_vars_init = tf.variables_initializer(self.all_saved, 'saved_init')
            self.filters_init = tf.variables_initializer(all_filters, 'filters_init')
            self.output = cur 


    def predict(self, sess, wave, beg, end, is_first):
        '''calculate top-level convolutions of a subrange of wave.
        if _is_first_run, previous state is re-initialized'''
        if (is_first):
            sess.run(self.saved_vars_init)
        ret = sess.run(self.output, feed_dict = { self.raw_input: wave[:,beg:end,:] })
        return ret

