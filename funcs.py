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
            n_out_chan,
            batch_sz):

        self.n_blocks = n_blocks
        self.n_block_layers = n_block_layers
        self.n_in_chan = n_in_chan
        self.n_out_chan = n_out_chan
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
            cur = self.raw_input

            # filters[b][l] = [pos][in_chan][out_chan]
            filters = [[None] * self.n_block_layers] * self.n_blocks

            # saved[b][l] = [batch][pos][out_chan]
            saved = [[None] * self.n_block_layers] * self.n_blocks
            all_saved = []

            filter_shape = [2, self.n_in_chan, self.n_out_chan]
            for b in range(self.n_blocks):
                with tf.name_scope('block%i' % b):
                    for l in range(self.n_block_layers):
                        with tf.name_scope('layer%i' % l):
                            dil = 2**l
                            filters[b][l] = self.create_filter('filter', filter_shape) 
                            saved_shape = [self.batch_sz, dil + 1, self.n_out_chan]
                            saved[b][l] = tf.Variable(tf.zeros(saved_shape), name='saved')
                            all_saved.append(saved[b][l])
                            conv = tf.nn.convolution(cur, filters[b][l], 'VALID')
                            cur = tf.concat([tf.stop_gradient(saved[b][l]), conv], 1, name='concat')
                            saved[b][l].assign(cur[:,-(dil+1):,:])

            self.saved_vars_init = tf.variables_initializer(all_saved, 'saved_init')
            self.output = conv


    def predict(self, sess, wave, beg, end, is_continue):
        '''calculate top-level convolutions of a subrange of wave.
        if _is_first_run, previous state is re-initialized'''
        if (not is_continue):
            sess.run(self.saved_vars_init)
        ret = sess.run(self.output, feed_dict = { self.raw_input: wave[:,beg:end,:] })
        return ret

