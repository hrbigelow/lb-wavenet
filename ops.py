import tensorflow as tf
import numpy as np

def mu_encode(x, n_quanta):
    '''mu-law encode and quantize'''
    mu = tf.to_float(n_quanta - 1)
    amp = tf.sign(x) * tf.log1p(mu * tf.abs(x)) / tf.log1p(mu)
    quant = (amp + 1) * 0.5 * mu + 0.5
    return tf.to_int32(quant)


def mu_decode(quant, n_quanta):
    '''accept an integer mu-law encoded quant, and convert
    it back to the pre-encoded value'''
    mu = tf.to_float(n_quanta - 1)
    qf = tf.to_float(quant)
    inv_mu = 1.0 / mu
    a = (2 * qf - 1) * inv_mu - 1
    x = tf.sign(a) * ((1 + mu)**abs(a) - 1) * inv_mu
    return x


def mu_encode_np(x, n_quanta):
    '''mu-law encode and quantize'''
    mu = n_quanta - 1
    amp = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quant = (amp + 1) * 0.5 * mu + 0.5
    return quant.astype(np.int32)


def mu_decode_np(quant, n_quanta):
    '''accept an integer mu-law encoded quant, and convert
    it back to the pre-encoded value'''
    mu = n_quanta - 1
    qf = quant.astype(np.float32)
    inv_mu = 1.0 / mu
    a = (2 * qf - 1) * inv_mu - 1
    x = np.sign(a) * ((1 + mu)**np.fabs(a) - 1) * inv_mu
    return x

def conv1x1(input, filt, batch_sz, name='conv1x1'):
    '''
    input: B x T x I
    filt: I x O
    output: B x T x O
    performs output[b][t] = filt * input[b][t] 
    '''
    #return tf.nn.convolution(input, tf.expand_dims(filt, 0), 'VALID', [1], [1], 'conv')

    with tf.name_scope(name):
        # filt_shape = tf.concat([[batch_sz], tf.shape(filt)], axis=0)
        fb = tf.stack([filt] * batch_sz)
        # fb = tf.broadcast_to(filt, filt_shape)
        conv = tf.matmul(input, fb)
    return conv

