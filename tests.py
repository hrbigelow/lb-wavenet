# Test 1: Whether tmodel and imodel are equivalent functions
import ops
import tensorflow as tf
import librosa
import numpy as np

def equiv_func(tmodel, imodel, data):
    '''tmodel: WaveNetTrain
       imodel: WaveNetGen
       data: MaskedSliceWave.wav_dataset
       '''



def mu(wav_in, sample_rate, n_quanta, wav_out):
    '''compare an original wav file to one
    that has been mu encoded and then decoded'''
    audio, _ = librosa.load(wav_in, sr=sample_rate, mono=True)
    audio_enc = ops.mu_encode(audio, n_quanta)
    audio_dec = ops.mu_decode(audio_enc, n_quanta)
    with tf.Session() as sess:
        audio_dec = sess.run(audio_dec)
    librosa.output.write_wav(wav_out, audio_dec, sample_rate)
    print('Sum squared error: ' + str(sum(pow(audio - audio_dec, 2))))


def get_tensor_sizes(sess):
    '''compute a summary of (name, num_elems, dtype)
    for all nodes in graph'''
    ind = []
    for op in sess.graph.get_operations():
        for ten in op.outputs:
            ind.append((ten.name, tf.shape(ten), tf.size(ten), ten.dtype))
    shape_ops = [e[1] for e in ind]
    size_ops = [e[2] for e in ind]
    sizes = sess.run(size_ops)
    shapes = sess.run(shape_ops)
    summary = [(i[0], z, np.array_str(s), i[3]) for i,z,s in zip(ind, sizes, shapes)]
    return summary


