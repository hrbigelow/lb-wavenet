# generate audio
import imodel
import tensorflow as tf
import librosa 
import json

batch_sz = 2 
chunk_sz = 100
sample_rate = 16000
log_dir = '/home/hrbigelow/ai/ckpt/lb-wavenet'
par_dir = '/home/hrbigelow/ai/par'
wav_dir = '/home/hrbigelow/ai/wav'

arch_file = 'arch3.json'
par_file = 'par1.json'
ckpt_file = 'arch3-1600' 

def main():

    with open(par_dir + '/' + arch_file, 'r') as fp:
        arch = json.load(fp)

    net = imodel.WaveNetGen(
            arch['n_blocks'],
            arch['n_block_layers'],
            arch['n_quant'],
            arch['n_res'],
            arch['n_dil'],
            arch['n_skip'],
            arch['n_post1'],
            arch['n_gc_embed'],
            arch['n_gc_category'],
            arch['use_bias'],
            batch_sz,
            chunk_sz)

    wave_op = net.build_graph()

    sess = tf.Session()
    net.restore(sess, log_dir + '/' + ckpt_file) 
    net.init_buffers(sess)

    gc_ids = [5,6]
    gen_sz = 100000 

    print('Starting inference...')
    i, wav_streams, wpos = sess.run(wave_op,
            feed_dict={
                net.gc_ids: gc_ids,
                net.gen_sz: gen_sz
                })

    print('Writing wav files.')
    for i in range(len(gc_ids)):
        path = '{}/gen.g{}.i{}.wav'.format(wav_dir, gc_ids[i], i)
        librosa.output.write_wav(path, wav_streams[i], sample_rate)
    print('Finished.')



if __name__ == '__main__':
    main()

