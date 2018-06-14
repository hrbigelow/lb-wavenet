import tensorflow as tf

import numpy as np

import funcs as wnet
wav = np.random.rand(1, 10000, 1)

g = tf.Graph()
net = wnet.WaveNet(1, 4, 1, 1, 1)
net.create_graph(g)
sess = tf.Session(graph=g)
sess.run(net.saved_vars_init)
sess.run(net.filters_init)



