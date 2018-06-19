# Tensorflow implementation of WaveNet

This is an implementation of WaveNet in TensorFlow, mostly as practice, and to get
any feedback from the community, though I hope it will also be useful to others.

## Approaches

I use the same caching approach as @tomlepaine/fast-wavenet for avoiding redundant
convolution calculations.  The cached values are stored in tf.Variable's so that the
values persist across sess.run() calls.

Another design principle is to use a single int32 placeholder tensor as a boolean value
which determines whether the computation graph will execute in 'initial' mode or 'continuation'
mode.  

 
