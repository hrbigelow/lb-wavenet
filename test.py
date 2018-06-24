import tensorflow as tf
import numpy as np
import funcs as wnet

n_blocks = 3
n_block_layers = 4
n_in_chan = 1
n_res_chan = 1
n_dil_chan = 1 
n_skip_chan = 1
n_post1_chan = 1
n_quant_chan = 1
batch_size = 1
wav = np.random.rand(batch_size, 100, n_in_chan)
net = wnet.WaveNet(n_blocks, n_block_layers, n_in_chan, n_res_chan, n_dil_chan,
        n_skip_chan, n_post1_chan, n_quant_chan)

g = net.create_graph()
sess = tf.Session(graph=g)
sess.run(net.saved_init, feed_dict = { net.raw_input: wav })
sess.run(net.filters_init)



