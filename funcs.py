import tensorflow as tf

def mu_encode(x, mu, n_quanta):
    '''mu-law encode and quantize'''
    mu_ten = tf.to_float(mu)
    amp = tf.sign(x) * tf.log1p(mu * tf.abs(x)) / tf.log1p(mu)
    quant = (amp + 1) * 0.5 * mu + 0.5


def create_filter(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable
    

def create_wavenet(n_blocks, n_block_layers, n_in_chan, n_out_chan, batch_sz):
    '''main wavenet structure'''
    raw = tf.placeholder(tf.float32, [batch_sz, None, n_in_chan])
    cur = raw
    # filters[b][l] = [pos][in_chan][out_chan]
    filters = [[None] * n_block_layers] * n_blocks

    # saved[b][l] = [batch][pos][out_chan]
    saved = [[None] * n_block_layers] * n_blocks
    all_saved = []

    for b in range(n_blocks):

        for l in range(n_block_layers):
            dil = 2**l
            filters[b][l] = create_filter('filter_%i_%i' % (b,l), [2, n_in_chan, n_out_chan]) 
            saved[b][l] = tf.Variable(tf.zeros([batch_sz, dil + 2, n_out_chan]),
                    name='saved_%i_%i' % (b,l))
            all_saved.append(saved[b][l])
            conv = tf.nn.convolution(cur, filters[b][l], 'VALID')
            cur = tf.concat([tf.stop_gradient(saved[b][l]), conv], 1)
            saved[b][l].assign(cur[:,-(dil+1):,:])

    saved_vars_init = tf.variables_initializer(all_saved)

    def init():
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())


    def predict(wave, beg, end, is_continue):
        '''calculate top-level convolutions of a subrange of wave.
        if _is_first_run, previous state is re-initialized'''
        with tf.Session() as sess:
            if (not is_continue):
                sess.run(saved_vars_init)
            sess.run(tf.global_variables_initializer())    
            return sess.run(conv, feed_dict = { raw: wave[:,beg:end,:] })

    return (init, predict)

