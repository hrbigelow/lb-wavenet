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

def parse_sample_map(sam_file):
    '''load a sample file with the format:
    voice_id /path/to/file1.wav
    voice_id /path/to/file2.wav
    ...
    '''
    sam_fh = open(sam_file)
    sample_map = [(int(t[0]), t[1]) for t in 
            [s.strip().split('\t') for s in 
                sam_fh.readlines()]]
    sam_fh.close()
    return sample_map 


def read_wav_file(wav, sample_rate):
    '''read the contents of a wav file, returning np.ndarray'''
    audio, _ = librosa.load(wav, sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    return audio


def concat_slices(itr, slice_sz, sess):
    '''generates slices from a virtually concatenated set of .wav files.
    itr produces { 'voice_id': int64 [], 'wav': int64 [None] } 

    yields { 'offsets': int64 [None,2], 'raw': int64 [slice_sz] }
    where:
    out['offsets'][i] = [voice_id, wav_offset, slice_offset]
    out['raw'][t] = wav_val
    '''
    need_sz = slice_sz 
    offsets = []
    spliced = tf.convert_to_tensor([], dtype=tf.int64)
    next_elem = itr.get_next()

    while True:
        try:
            wav = sess.run(next_elem)
        except tf.errors.OutOfRangeError:
            break
        raw = wav['raw']
        vid = wav['vid']
        cur_item_pos = 0
        
        while need_sz <= (item_sz - cur_item_pos):
            spliced = tf.concat([spliced, raw[cur_item_pos:cur_item_pos + need_sz]], 0)
            offsets += [vid, cur_item_pos, slice_sz - need_sz]
            cur_item_pos += need_sz 
            yield { 'offsets': tf.convert_to_tensor(offsets), 'spliced': spliced } 
            offsets = [] 
            spliced = tf.convert_to_tensor([], dtype=tf.int64)
            need_sz = slice_sz 
        if cur_item_pos != item_sz:
            spliced = tf.concat([spliced, raw[cur_item_pos:]], 0) 
            offsets += [vid, cur_item_pos, sliced_sz - need_sz] 
            need_sz -= item_sz - cur_item_pos
    return


def gen_wav_files(path_itr, sample_rate, sess):
    next_el = path_itr.get_next()
    while True:
        try:
            path = sess.run(next_el)
        except tf.errors.OutOfRangeError:
            break
        wav = read_wav_file(path['wav'], sample_rate)
        yield (path['voice_id'], wav)
    return


def wav_concat_slices(sam_path, slice_sz, sample_rate, sess):
    '''parse a sample file and create a ts.data.Dataset of concatenated,
    labeled slices from it'''
    samples = parse_sample_map(sam_path)
    d1 = tf.data.Dataset.from_tensor_slices({ 
        'voice_id': [i[0] for i in samples],
        'wav': [i[1] for i in samples]
        })
    d2 = d1 # do shuffling and repetition here
    d3 = tf.data.Dataset.from_generator(gen_wav_files,
            (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])),
            (d2, sample_rate, sess))
    out_types = (tf.int64, tf.int64)
    out_shapes = (tf.TensorShape([None]), tf.TensorShape([None]))
    d4 = tf.data.Dataset.from_generator(concat_slices, out_types, out_shapes,
            (d3, slice_sz, sess))
    return d4



def gen1(lim):
    for i in range(lim):
        yield i
    return


def gen2(itr, sess):
    next_el = itr.get_next()
    while True:
        try:
            v = sess.run(next_el)  
        except tf.errors.OutOfRangeError:
            break
        #vi = sess.run(v)
        #vt = tf.convert_to_tensor(vi)
        yield v 
    return



ds1 = tf.data.Dataset.from_generator(lambda : gen1(10), (tf.int32), (tf.TensorShape([])))
itr1 = ds1.make_one_shot_iterator()

sess = tf.Session()
ds2 = tf.data.Dataset.from_generator(lambda : gen2(itr1, sess), (tf.int32), (tf.TensorShape([])))
itr2 = ds2.make_one_shot_iterator()



        
