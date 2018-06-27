# functions for preprocessing and loading data

# parse a user-provided list of ID's and path names to .wav files
# use the list to construct the input
# perhaps it's better to abstract away the function for reading a file
# from that used to extract a unit of

# decision
# 1. parse the main file into a map
# 2. enqueue the files
# 3. load them as needed into empty slots. 

# fuctions needed

import librosa
import tensorflow as tf
import numpy as np

def gen_sample_map(sam_file):
    '''load a sample file with the format:
    voice_id /path/to/file1.wav
    voice_id /path/to/file2.wav
    ...
    generate tuples (voice_id, wav_path)
    '''
    with open(sam_file) as sam_fh:
        for s in sam_fh.readlines():
            (vid, wav_path) = s.strip().split('\t')
            yield (int(vid), wav_path)
    return


def read_wav_file(wav_file, sample_rate):
    '''read the contents of a wav file, returning np.ndarray'''
    audio, _ = librosa.load(wav_file, sr=sample_rate, mono=True)
    #audio = audio.reshape(-1, 1)
    return audio


def gen_wav_files(path_itr, sample_rate, sess):
    '''consume an iterator that yields [voice_id, wav_path].
    load the .wav file contents into a vector and return a tuple
    '''
    ne = path_itr.get_next()
    while True:
        try:
            (vid, wav_path) = sess.run(ne)
        except tf.errors.OutOfRangeError:
            break
        wav = read_wav_file(wav_path, sample_rate)
        yield (int(vid), wav)
    return


def gen_concat_slices(wav_itr, slice_sz, recep_field_sz, sess):
    '''generates slices from a virtually concatenated set of .wav files.
    consume itr, which yields [voice_id, wav_data]

    concatenate slices of slice_sz, yielding 
    (int32 [slice_sz], float32 [slice_sz])
    where:
    out[0][t] = wav_val
    out[1][t] = mid 
    out[2][mid] = vid

    vid is the voice id of the speaker.  mid is the mapping id used for this
    slice.
    
    the special id value of zero indicates that this position is an invalid
    training window.
    '''
    need_sz = slice_sz 
    spliced_wav = np.empty(0, np.float)
    spliced_ids = np.empty(0, np.int32)
    ne = wav_itr.get_next()
    zero_lead = np.full(recep_field_sz - 1, 0, np.int32)
    idmap = [0]

    while True:
        try:
            (vid, wav) = sess.run(ne)
        except tf.errors.OutOfRangeError:
            break
        wav_sz = wav.shape[0] 
        if wav_sz < recep_field_sz:
            printf('Warning: skipping length %i wav file (voice id %i).  '
                    + 'Shorter than receptive field size of %i\n' 
                    % (wav_sz, vid, recep_field_sz))
            continue
        try:
            mid = idmap.index(vid)
        except ValueError:
            idmap += vid
            mid = len(idmap) - 1

        slice_hi_bound = min(need_sz, wav_sz)
        slice_lo_bound = max(need_sz, recep_field_sz - 1)
        ids = np.concatenate([
            zero_lead, 
            np.full(slice_hi_bound - slice_lo_bound, mid, np.int32), 
            np.full(wav_sz - slice_hi_bound, 1, np.int32)
            ])

        cur_item_pos = 0
        
        while need_sz <= (wav_sz - cur_item_pos):
            # use up a chunk of the current item and yield the slice
            spliced_wav = np.append(spliced_wav, wav[cur_item_pos:cur_item_pos + need_sz])
            spliced_ids = np.append(spliced_ids, ids[cur_item_pos:cur_item_pos + need_sz])
            cur_item_pos += need_sz 
            yield (spliced_wav, spliced_ids, idmap) 
            spliced_wav = np.empty(0, np.float) 
            spliced_ids = np.empty(0, np.int32)
            idmap = [0, vid]
            need_sz = slice_sz 
        if cur_item_pos != wav_sz:
            # append this piece of wav to the current slice 
            spliced_wav = np.append(spliced_wav, wav[cur_item_pos:])
            spliced_ids = np.append(spliced_ids, ids[cur_item_pos:])
            need_sz -= (wav_sz - cur_item_pos)
    return



def wav_dataset(sam_path, slice_sz, batch_sz, recep_field_sz, sample_rate, sess):
    '''parse a sample file and create a ts.data.Dataset of concatenated,
    labeled slices from it'''
    zero_d = tf.TensorShape([])
    one_d = tf.TensorShape([None])
    d1 = tf.data.Dataset.from_generator(lambda: gen_sample_map(sam_path),
            (tf.int32, tf.string),
            (zero_d, zero_d))

    d2 = d1 # do shuffling and repetition here
    i2 = d2.make_one_shot_iterator()

    d3 = tf.data.Dataset.from_generator(lambda: gen_wav_files(i2, sample_rate, sess),
            (tf.int32, tf.float32), (zero_d, one_d))
    i3 = d3.make_one_shot_iterator()

    d4 = tf.data.Dataset.from_generator(
            lambda: gen_concat_slices(i3, slice_sz, recep_field_sz, sess),
            (tf.float32, tf.int32, tf.int32), (one_d, one_d, one_d))

    slice_ne = [d4.make_one_shot_iterator().get_next()] * batch_sz
    def next_slice():
        return sess.run(slice_ne)

    return next_slice 


def 
