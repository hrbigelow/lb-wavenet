import tensorflow as tf
import numpy as np
import funcs as wnet

n_blocks = 5
n_block_layers = 10
n_in_chan = 5
n_res_chan = 16
n_dil_chan = 32
batch_size = 10
wav = np.random.rand(batch_size, 10000, n_in_chan)
net = wnet.WaveNet(n_blocks, n_block_layers, n_in_chan, n_res_chan, n_dil_chan)

g = tf.Graph()
net.create_graph(g)
sess = tf.Session(graph=g)
sess.run(net.saved_init)
sess.run(net.filters_init)



