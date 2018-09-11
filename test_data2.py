import itertools
import tensorflow as tf

tf.enable_eager_execution()

def gen():
    strings = ['abc', 'bcd', 'cde', 'def']
    for s in strings:
        yield bytes(s, 'utf-8') 
    return

zero_d = tf.TensorShape([])

ds = tf.data.Dataset.from_generator(gen, (tf.string), (zero_d))
it = ds.make_one_shot_iterator()

while True:
    try:
        t = it.get_next()
        print(t)
    except:
        break
