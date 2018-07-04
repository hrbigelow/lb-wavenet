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

class MaskedSliceWav(object):

    def __init__(self,
            sam_file,
            batch_sz,
            sample_rate, 
            slice_sz,
            recep_field_sz
            ):
        self.sam_file = sam_file
        self.batch_sz = batch_sz
        self.sample_rate = sample_rate
        self.slice_sz = slice_sz
        self.recep_field_sz = recep_field_sz
        
    def init_sample_catalog(self):
        self.sample_catalog = []
        with open(self.sam_file) as sam_fh:
            for s in sam_fh.readlines():
                (vid, wav_path) = s.strip().split('\t')
                self.sample_catalog.append([int(vid), wav_path])


    def _gen_sample_map(self):
        '''load a sample file with the format:
        voice_id /path/to/file1.wav
        voice_id /path/to/file2.wav
        ...
        generate tuples (voice_id, wav_path)
        '''
        for s in self.sample_catalog:
            (vid, wav_path) = (s[0], s[1])
            yield (vid, wav_path)
        return


    def _read_wav_file(self, wav_file):
        '''read the contents of a wav file, returning np.ndarray'''
        audio, _ = librosa.load(wav_file, sr=self.sample_rate, mono=True)
        #audio = audio.reshape(-1, 1)
        return audio


    def _gen_wav_files(self, path_itr, sess):
        '''consume an iterator that yields [voice_id, wav_path].
        load the .wav file contents into a vector and return a tuple
        '''
        next_el = path_itr.get_next()
        while True:
            try:
                (vid, wav_path) = sess.run(next_el)
            except tf.errors.OutOfRangeError:
                break
            wav = self._read_wav_file(wav_path)
            yield (int(vid), wav)
        return


    def _gen_concat_slices(self, wav_itr, sess):
        '''generates slices from a virtually concatenated set of .wav files.
        consume itr, which yields [voice_id, wav_data]

        concatenate slices of self.slice_sz, yielding 
        where:
        wav[t] = wav_val
        ids[t] = mid 
        idmap[mid] = vid

        vid is the voice id of the speaker.  mid is the mapping id used for this
        slice.
        
        the special id value of zero indicates that this position is an invalid
        training window.
        '''
        need_sz = self.slice_sz 
        spliced_wav = np.empty(0, np.float)
        spliced_ids = np.empty(0, np.int32)
        next_el = wav_itr.get_next()
        recep_bound = self.recep_field_sz - 1
        idmap = [0]

        while True:
            try:
                (vid, wav) = sess.run(next_el)
            except tf.errors.OutOfRangeError:
                break
            wav_sz = wav.shape[0] 
            if wav_sz < self.recep_field_sz:
                printf('Warning: skipping length %i wav file (voice id %i).  '
                        + 'Shorter than receptive field size of %i\n' 
                        % (wav_sz, vid, self.recep_field_sz))
                continue
            try:
                mid = idmap.index(vid)
            except ValueError:
                idmap += vid
                mid = len(idmap) - 1

            slice_bound = min(need_sz, wav_sz)
            mid_bound = max(recep_bound, slice_bound)
            
            ids = np.concatenate([
                np.full(recep_bound, 0, np.int32),
                np.full(mid_bound - recep_bound, mid, np.int32), 
                np.full(wav_sz - mid_bound, 1, np.int32)
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
                need_sz = self.slice_sz 
            if cur_item_pos != wav_sz:
                # append this piece of wav to the current slice 
                spliced_wav = np.append(spliced_wav, wav[cur_item_pos:])
                spliced_ids = np.append(spliced_ids, ids[cur_item_pos:])
                need_sz -= (wav_sz - cur_item_pos)
        return



    def wav_dataset(self, sess):
        '''parse a sample file and create a ts.data.Dataset of concatenated,
        labeled slices from it'''
        zero_d = tf.TensorShape([])
        one_d = tf.TensorShape([None])
        with tf.name_scope('dataset'):
            with tf.name_scope('sample_map'):
                d1 = tf.data.Dataset.from_generator(
                        self._gen_sample_map,
                        (tf.int32, tf.string),
                        (zero_d, zero_d))

            with tf.name_scope('arranging'):
                d2 = d1 # do shuffling and repetition here
                i2 = d2.make_one_shot_iterator()

            with tf.name_scope('reading_files'):
                d3 = tf.data.Dataset.from_generator(
                        lambda: self._gen_wav_files(i2, sess),
                        (tf.int32, tf.float32), (zero_d, one_d))
                i3 = d3.make_one_shot_iterator()

            with tf.name_scope('slicing'):
                d4 = tf.data.Dataset.from_generator(
                        lambda: self._gen_concat_slices(i3, sess),
                        (tf.float32, tf.int32, tf.int32), (one_d, one_d, one_d))

            with tf.name_scope('batching'):
                slices = [d4.make_one_shot_iterator().get_next()] * self.batch_sz
                wav = tf.stack([s[0] for s in slices])
                ids = [s[1] for s in slices]
                id_maps = [s[2] for s in slices]

        return [wav, ids, id_maps] 
