# Tensorflow implementation of WaveNet

This is an implementation of WaveNet in TensorFlow, mostly as practice, and to get
any feedback from the community, though I hope it will also be useful to others.

## Conditional independence across time boundary

WaveNet's stack of dilated convolutions means all nodes to the right of a time
boundary are conditionally independent from all nodes to the left, given a
subset of "D-separation" nodes (see diagram).  Because of this, the entire
output of an arbitrarily long input sequence can be broken up in stages.  At
each stage, we prepend each convolution with the previously stored D-separation
node values.  At the end, we store the last convolution values for use in the
next stage.  This approach is the same as used by @tomlepaine.

Note also that all convolutions are 'VALID', so the output is shorter than the
input by the dilation amount.  But, the number of prepended D-separation node
values is also equal to this dilation amount, so each output convolution is the
same length as the input before prepending.  In this way, the continuation
produces the same values as a virtual slice of an indefinitely long
computation.


## SGD batches use consecutive samples from multiple .wav files 

For supervised training, each (x, y) sample consists of a window of values from
the .wav file as x, and the next value following the window as the target y.
The size of the window equals the receptive field F of one output node, equal
to n_blocks * sum([2**l for l in range(n_block_layers)]).  So, a .wav file of
length N will provide N - F - 1 training samples.  Two samples whose windows
are offset by 1 in the .wav file are called 'consecutive samples'.

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
the first F-1 windows at the start of training are considered to span a
boundary between .wav files and thus marked invalid.  The invalid windows will
in general occur in different positions in each of the B slots, but this
doesn't pose any problem. 

This scheme incurs a fraction of wasted calculation equal to F / W, where W is
the average .wav file length.  For VCTK and a network with F = 3000, this is
roughly 5 percent.  This waste is avoidable, but the only way to avoid it and
still maintain B parallel computations is to arrange the .wav files' boundaries
to line up, and then use a separate code path that avoids prepending
D-separation values.  The point is basically moot, because there is no way in
general to force .wav file boundaries to line up.  My decision is to incur the
wasted calculation and keep the pipeline simple.


## Saved D-separation values allow continuation through invalid sample windows

![Influence of Nodes](images/wavenet_influence.png)

The diagram above illustrates how the D-separation mechanism works even in the
presence of invalid sample windows (i.e. windows of input that span an
artificial boundary between two concatenated .wav files).

At the bottom of the diagram is the input, showing the end of one .wav file as
'0' values, and the beginning of another as '4096' values.  The 0 and 4096
valued input is understood to extend indefinitely, to the left and right,
respectively. 

In this set of convolutions of dilations [1, 2, 4, 8] * 3, node values are
shown and color coded, using all filter weights of [0.5, 0.5].  The choice of
filters is meant to illustrate that any nodes that are not either zero-valued
or 4096-valued are resulting from an artificial, junction-spanning sample
window.  

However, even if the stage boundary is as shown (vertical black line), the
continued stage which uses the D-separation values outlined in purple will
start to produce valid results as soon as the receptive field moves beyond the
junction.  This can be seen by the output nodes that show a value of 4096
towards the right of the diagram.


