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
    for all nodes in graph:
    summary[i] = (name, size, shape_string, dtype)'''
    summary = []
    for op in sess.graph.get_operations():
        for ten in op.outputs:
            try:
                item = (ten.name, str(ten.shape.as_list()),
                ten.shape.num_elements(), ten.dtype.size)
            except ValueError:
                item = (ten.name, '?', '?', '?')

            summary.append(item)
    return summary


def test_dataset(sess, **kwargs):
    import data
    dset = data.MaskedSliceWav(
            kwargs['sam_file'],
            kwargs['batch_sz'],
            kwargs['sample_rate'],
            kwargs['slice_sz'],
            kwargs['prefetch_sz'],
            kwargs['recep_field_sz']
            )
    dset.init_sample_catalog()
    print('Creating dataset...', end='', flush=True)
    dset_ops = dset.wav_dataset(sess)
    print('done')
    print('Running ops {}...'.format(str(dset_ops)), end='', flush=True)
    wav, mask, maps = sess.run(dset_ops)
    print('done')
    print(len(wav))

