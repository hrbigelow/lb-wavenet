# Tensorflow implementation of WaveNet

This is an implementation of WaveNet in TensorFlow, mostly as practice, and to get
any feedback from the community, though I hope it will also be useful to others.

## Conditional independence across time boundary

WaveNet's stack of dilated convolutions means all nodes to the right of a time
boundary are conditionally independent from all nodes to the left, given a
subset of "D-separation" nodes.  Below, the boundary is marked in black, and
the D-separation nodes are outlined in purple.  Because of this, the entire
output of an arbitrarily long input sequence can be broken up in stages.  At
each stage, we prepend each convolution with the previously stored D-separation
node values.  At the end, we store the last convolution values for use in the
next stage.

## Samples from .wav files

For supervised training, each (x, y) sample consists of a window of values from
the .wav file as x, and the next value following the window as the target y.
The size of the window equals the receptive field F of one output node, equal
to n_blocks * sum([2**l for l in range(n_block_layers)]).  So, a .wav file of
length N will provide N - F - 1 training samples.  Two samples whose windows
are offset by 1 in the .wav file are called 'consecutive samples'.

## Splitting up continuous input using saved D-separation values 

As a compromise between efficiency and diversity of samples for each SGD batch,
we compute a number of consecutive samples for some number B of .wav files all
at once, and use them in the SGD batch.  Overall, the entire set of .wav files
are concatenated end-to-end in B groups, and fed into the network, and then
processed in stages using the D-separation continuation mechanism.

# Marking invalid sample windows

The end-to-end concatenation of different .wav files produces invalid windows.
Instead of padding with zeroes or breaking up the pipeline, the network reads
right through these and computes output.  It keeps track of where the
boundaries are, and marks any sample windows that span a boundary as invalid,
and then excludes those from the SGD gradient calculation.  Logically speaking,
the first F windows at the start of training all can be considered to span a
boundary between .wav files.   

## D-separation values have no influence beyond receptive field

From stage to stage, saved D-separation values may be the result of invalid
windows.  However, this does not interfere with their ability to propagate the
appropriate information to later stages to obtain valid calculations.  In this
way, the continuation mechanism is independent of the mechanism for marking
invalid sample windows.

This scheme incurs a fraction of wasted calculation equal to F / W_avg, where
W_avg is the average .wav file length.  For VCTK, this is roughly 5 percent.
Is this avoidable?  It is, but the only way to avoid this waste and still
maintain B parallel computations is to arrange the .wav files' boundaries to
line up, which is in general impossible.  My decision is to incur the wasted
calculation and keep the pipeline simple.


## Saved intermediate values allow continuation through invalid sample windows

The diagram below also illustrates how input nodes influence the output.  The
raw input is shown on the bottom row with dilated convolutions [1, 2, 4, 8] * 3
shown above, the top row being the final output.  The input values are zeros
extending infinitely to the left, and 4096 extending to the right, with a
boundary near the left end of the diagram.  All filter weights are set to
\[0.5, 0.5\].  In this setup, any given convolution node reflects the fraction
of zero and 4096-valued input nodes.  Notice that the top row nodes are zero up
until the point where the input transition starts.  Also, the left-most
4096-valued output node (top, towards the right) is the first position where
the receptive field does not cross the boundary in the input.  All output nodes
in between reflect a mix of influence from zero-valued and 4096-valued input.

Consider the convolutions extending indefinitely in either direction as shown.
At any given boundary (such as the black vertical line, the values of the set
of all nodes to the right only depend on the values of the nodes outlined in
purple.  All other nodes to the left merely propagate their influence via those
in purple. 



![Influence of Nodes](images/wavenet_influence.png)

I use the same caching approach as @tomlepaine fast-wavenet for avoiding redundant
convolution calculations.  The cached values are stored in tf.Variable's so that the
values persist across sess.run() calls.


## Computation graph configuration tensors

Both batch size and input length are unspecified in the graph.  Mode ('initial'
or 'continuation') is specified using a single 'is_first' int32 placeholder.
The tf.Variables used to store intermediate convolution values are fixed in
shape.  But, when run in 'initial' mode, the is_first value is used to
determine whether to prepend a zero-length slice or the full length slice of
these stored values. 


The very first SGD batch includes the first samples from each .wav file.  For
example with VCTK files, a short period of background noise before the speaker
starts the sentence.  This first batch will be run in 'initial' mode.
Subsequent SGD batches will be run on the next window of input
(non-overlapping) using 'continuation' mode.  Because of the structure of the
network, the computed output values in continuation mode will be identical to
those that would be produced in initial mode with a longer window.

