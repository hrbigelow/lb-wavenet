# Tensorflow implementation of WaveNet

This is an implementation of WaveNet in TensorFlow, mostly as practice, and to get
any feedback from the community, though I hope it will also be useful to others.

## Approaches

I use the same caching approach as @tomlepaine fast-wavenet for avoiding redundant
convolution calculations.  The cached values are stored in tf.Variable's so that the
values persist across sess.run() calls.

The chain of dilated convolutions are always stride=1 and VALID, so each output
tensor is shorter than the previous by the size of the dilation.  While it is
typical to think of the 'receptive field' of a single output node.  In this
architecture, the receptive fields of any two adjacent output nodes are two
windows of input nodes offset by one position.  It is useful, then, to think of
'the receptive field of a window of output nodes as a window of input.  The
relationship is:

```
output_sz = input_sz - burn_sz 
burn_sz = n_blocks * sum([2**l for l in range(n_block_layers)]).  
```

Note also that in computing windows of output at a time avoids redundant
calculations.  The redundant calculations only arise when one must break up the
input into certain sized lengths, either to compute a batch of gradients or to
fit within a certain memory.

lb-wavenet (lb = 'lookback') has two modes during training: 'initial mode' and
'continuation mode'.  In initial mode, it consumes some window of input and
produces a window of output shorter by burn_sz.  In continuation mode, it
consumes input and uses cached values of previous time steps for each
convolution (and input) so that it can consume input starting just after where
it left off.

In the animation below, hollow circles represent nodes that haven't been read
or calculated.  Solid represents read or calculated.  Red nodes are those
values stored in a variable to be read at the next execution of the computation
graph.  
   

## Computation graph configuration tensors

Both batch size and input length are unspecified in the graph.  Mode ('initial'
or 'continuation') is specified using a single 'is_first' int32 placeholder.
The tf.Variables used to store intermediate convolution values are fixed in
shape.  But, when run in 'initial' mode, the is_first value is used to
determine whether to prepend a zero-length slice or the full length slice of
these stored values. 


## Batching and different length input .wav files

From the point of view of WaveNet, a single .wav file of length N represents N
- burn_sz individual training samples, each one a window of burn_sz + 1 input
  values, with the next value as the target.  For a chosen SGD batch size of
samples, one might ask whether the optimal batch would consist of scattered
windows from the files, or if it is okay to have a number of consecutive
windows from the same .wav file.  From a computational efficiency point of
view, insisting on completely scattered windows would be extremely inefficient,
requiring full computation of convolutions starting from the entire burn_sz
window.

In this work, I have taken the approach to require consecutive windows of input
in each SGD batch, but to allow easily adjusting the number of windows and
number of different individual wav files.


 

