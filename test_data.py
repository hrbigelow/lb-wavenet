import data
import json
import tensorflow as tf

sam_file = '/home/hrbigelow/ai/data/ljspeech.rdb'
par_path = '/home/hrbigelow/ai/par/par1.json'
arch_path = '/home/hrbigelow/ai/par/arch4.json'
recep_field_sz = 10000

def main():

    with open(par_path, 'r') as fp:
        par = json.load(fp)

    with open(arch_path, 'r') as fp:
        arch = json.load(fp)

    dset = data.MaskedSliceWav(
            sam_file,
            recep_field_sz,
            par['sample_rate'],
            par['slice_sz'],
            par['prefetch_sz'],
            arch['n_lc_in'],
            arch['lc_hop_sz'],
            par['batch_sz']
            )

    dset.init_sample_catalog()

    #zero_d = tf.TensorShape([])
    #ds = tf.data.Dataset.from_generator(
    #        dset._gen_path,
    #        (tf.int32, tf.string, tf.string),
    #        (zero_d, zero_d, zero_d))
    #itr = ds.make_one_shot_iterator()
    #ops = itr.get_next()
    #return ops

    sess = tf.Session()
    ops = dset.wav_dataset(sess)

    l = 0
    while True:
        try:
            wav, mel, mask = sess.run(ops)
            print(l, len(wav[0]), len(mel[0]), len(mask[0]))
            l += 1
        except tf.errors.OutOfRangeError:
            print('Reached end of data set')
            break
    return


if __name__ == '__main__':
    main()

