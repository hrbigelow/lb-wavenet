import tensorflow as tf

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

