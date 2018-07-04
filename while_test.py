import tensorflow as tf

i = tf.constant(2)
n = 100
a = tf.TensorArray(dtype=tf.int32, size = n, clear_after_read = False)

# initial values for fibonacci sequence
a = a.write(0, 1)
a = a.write(1, 1)

def fib(p, q):
    return tf.add(p, q)

def p(i, vals):
    return i < n

def body(i, vals):
    a = vals.read(i - 2)
    b = vals.read(i - 1)
    f = fib(a, b)
    vals = vals.write(i, f)
    return i + 1, vals


(ret_i, ret_a) = tf.while_loop(p, body, [i, a], back_prop = False, parallel_iterations = 1)


