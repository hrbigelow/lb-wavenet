# generate audio
import imodel
import tensorflow as tf
import librosa 
import json

batch_sz = 10
chunk_sz = 1000
sample_rate = 16000
log_dir = '/home/hrbigelow/ai/ckpt'
par_dir = '/home/hrbigelow/ai/par'

arch_file = 'arch1.json'
par_file = 'par1.json'
ckpt_file = 'arch1-100' 

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
            batch_sz,
            chunk_sz)

    wave_op = net.build_graph()

    sess = tf.Session()
    net.restore(sess, log_dir + '/' + ckpt_file) 
    gc_ids = [5,8,10]
    gen_sz = 10000

    wav_streams = sess.run(wave_op,
            feed_dict={net.gc_ids: gc_ids, net.gen_sz: gen_sz})



if __name__ == '__main__':
    main()

