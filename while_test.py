import tensorflow as tf

n = 10
a = tf.TensorArray(dtype=tf.int32, size = n, clear_after_read = False)
a = a.write(0, 1)
a = a.write(1, 1)

def cond(i):
    return i < n

def body(i):
    a = a.write(i, a.read(i - 2) + a.read(i - 1))
    return i + 1


ret_i = tf.while_loop(cond, body, [2])

#sess = tf.Session()
#sess.run(a.stack())

