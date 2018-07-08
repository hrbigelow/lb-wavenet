import tensorflow as tf

v = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
n = len(v)
a = tf.Variable(v, name = 'a')

def cond(i, a):
    return i < n 

def body(i, a):
    a = tf.assign(a[i], a[i-1] + a[i-2])
    return i + 1, a

i, b = tf.while_loop(cond, body, [2, a]) 

