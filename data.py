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


    def _gen_path(self):
        '''load a sample file with the format:
        voice_id /path/to/file1.wav
        voice_id /path/to/file2.wav
        ...
        generate tuples (voice_id, wav_path)
        '''
        for s in self.sample_catalog:
            vid, wav_path = s[0], s[1]
            yield vid, wav_path
        return


    def _read_wav_file(self, wav_file):
        '''read the contents of a wav file, returning np.ndarray'''
        audio, _ = librosa.load(wav_file, sr=self.sample_rate, mono=True)
        print('Reading %s' % wav_file)
        #audio = audio.reshape(-1, 1)
        return audio


    def _wav_gen(self, path_itr, sess):
        '''consume an iterator that yields [voice_id, wav_path].
        load the .wav file contents into a vector and return a tuple
        generate tuples (voice_id, [wav_val, wav_val, ...])
        '''
        next_el = path_itr.get_next()
        while True:
            try:
                vid, wav_path = sess.run(next_el)
                wav = self._read_wav_file(wav_path)
                yield int(vid), wav
            except tf.errors.OutOfRangeError:
                break
        return


    def _gen_concat_slice_factory(self, wav_gen):
        '''factory function for creating a new generator
        wav_gen: generates next (vid, wav) pair
        '''

        def gen_fcn():
            '''generates a slice from a virtually concatenated set of .wav files.
            consume itr, which yields [voice_id, wav_data]

            concatenate slices of self.slice_sz where:
            spliced_wav[t] = wav_val
            spliced_ids[t] = mapping_id 
            idmap[mapping_id] = voice_id

            mapping_id corresponds to voice_id for valid positions, or zero for invalid
            (positions corresponding to junction-spanning receptive field windows)

            generates (spliced_wav, spliced_ids, idmap)
            '''
            need_sz = self.slice_sz 
            spliced_wav = np.empty(0, np.float)
            spliced_ids = np.empty(0, np.int32)
            recep_bound = self.recep_field_sz - 1
            idmap = np.array([0], np.int32)

            while True:
                try:
                    vid, wav = next(wav_gen) 
                except StopIteration:
                    break
                wav_sz = wav.shape[0] 
                if wav_sz < self.recep_field_sz:
                    printf('Warning: skipping length %i wav file (voice id %i).  '
                            + 'Shorter than receptive field size of %i\n' 
                            % (wav_sz, vid, self.recep_field_sz))
                    continue
                try:
                    mid = idmap.tolist().index(vid)
                except ValueError:
                    idmap = np.append(idmap, vid)
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
                    # print(str(need_sz) + ', ' + str(wav_sz - cur_item_pos))
                    # use up a chunk of the current item and yield the slice
                    spliced_wav = np.append(spliced_wav, wav[cur_item_pos:cur_item_pos + need_sz])
                    spliced_ids = np.append(spliced_ids, ids[cur_item_pos:cur_item_pos + need_sz])
                    cur_item_pos += need_sz 
                    yield spliced_wav, spliced_ids, idmap 
                    spliced_wav = np.empty(0, np.float) 
                    spliced_ids = np.empty(0, np.int32)
                    idmap = np.array([0, vid], np.int32)
                    need_sz = self.slice_sz 

                if cur_item_pos != wav_sz:
                    # append this piece of wav to the current slice 
                    spliced_wav = np.append(spliced_wav, wav[cur_item_pos:])
                    spliced_ids = np.append(spliced_ids, ids[cur_item_pos:])
                    need_sz -= (wav_sz - cur_item_pos)
            return
        return gen_fcn


    def _gen_slice_batch(self, path_itr, sess):
        '''generates a batch of concatenated slices
        yields:
        wav[b][t] = amplitude
        ids[b][t] = mid or zero (mask)
        idmap[b][mid] = vid
        b = batch, t = timestep
        '''
        # construct the single (vid, wav) generator
        wav_gen = self._wav_gen(path_itr, sess)

        # construct batch_sz slice generators, each sharing the same wav_gen
        gens = [self._gen_concat_slice_factory(wav_gen)() for _ in range(self.batch_sz)]

        while True:
            try:
                batch = [next(g) for g in gens]
                wav = np.stack([b[0] for b in batch])
                ids = np.stack([b[1] for b in batch])
                idmaps = [b[2] for b in batch]
                idmaps_len = [m.shape[0] for m in idmaps]
                ml = max(idmaps_len)

                # pad each with zeros at the end so we can stack
                pairs = zip(idmaps, idmaps_len)
                idmaps_pad = [np.pad(m, ((0, ml - l),), mode='constant') for m,l in pairs]
                idmap = np.stack(idmaps_pad)
                yield wav, ids, idmap 
            except StopIteration:
                # this will be raised if wav_itr runs out
                break


    def wav_dataset(self, sess):
        '''parse a sample file and create a ts.data.Dataset of concatenated,
        labeled slices from it.
        d1 items are (voice_id, wav_path)
        d2 items are just d1 items shuffled and repeated
        d3 items are (voice_id, [wav_val, wav_val, ...])
        d4 items are (spliced_wav, spliced_ids, idmap)

        returns:
            wav: ['''
        zero_d = tf.TensorShape([])
        two_d = tf.TensorShape([self.batch_sz, None])

        with tf.name_scope('dataset'):
            with tf.name_scope('sample_map'):
                path_ds = tf.data.Dataset.from_generator(
                        self._gen_path,
                        (tf.int32, tf.string),
                        (zero_d, zero_d))

            with tf.name_scope('shuffle_repeat'):
                path_shuf_ds = path_ds.shuffle(buffer_size=len(self.sample_catalog))
                path_shuf_itr = path_shuf_ds.make_one_shot_iterator()

            with tf.name_scope('slice_batch'):
                slice_batch_ds = tf.data.Dataset.from_generator(
                        lambda: self._gen_slice_batch(path_shuf_itr, sess),
                        (tf.float32, tf.int32, tf.int32),
                        (two_d, two_d, two_d))
                slice_batch_itr = slice_batch_ds.make_one_shot_iterator()

            wav, ids, idmap = slice_batch_itr.get_next()
        return wav, ids, idmap
            

