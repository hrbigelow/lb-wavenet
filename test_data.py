import data
import json
import tensorflow as tf

sam_file = '/home/hrbigelow/ai/data/vctk_samples.rdb'
par_path = '/home/hrbigelow/ai/par/par1.json'
recep_field_sz = 10000

def main():

    with open(par_path, 'r') as fp:
        par = json.load(fp)

    dset = data.MaskedSliceWav(
            sam_file,
            par['batch_sz'],
            par['sample_rate'],
            par['slice_sz'],
            recep_field_sz
            )

    dset.init_sample_catalog()
    sess = tf.Session()
    wav_input, id_masks, id_maps = dset.wav_dataset(sess)

    while True:
        try:
            wav, masks, maps = sess.run([wav_input, id_masks, id_maps])
            # print(masks)
        except tf.errors.OutOfRangeError:
            print('Reached end of data set')
            break
    return


if __name__ == '__main__':
    main()

