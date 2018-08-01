# generate audio
import imodel
import tensorflow as tf
import librosa 
import json
from tensorflow.python import debug as tf_debug

batch_sz = 2 
chunk_sz = 100
sample_rate = 16000
log_dir = '/home/hrbigelow/ai/ckpt/lb-wavenet'
tb_dir = '/home/hrbigelow/ai/tb/lb-wavenet'
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

    print('Building graph.')
    wave_op = net.build_graph()

    sess = tf.Session()
    ckpt_path = '{}/{}'.format(log_dir, ckpt_file)
    print('Restoring from {}'.format(ckpt_path))
    net.restore(sess, ckpt_path) 

    print('Creating FileWriter.')
    fw = tf.summary.FileWriter(tb_dir, graph=sess.graph)

    print('Initializing buffers.')
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'asus:6064')
    net.init_buffers(sess)

    gc_ids = [5,6]
    gen_sz = 50000 
    feed_dict = { net.gen_sz: gen_sz }
    try:
        feed_dict[net.gc_ids] = gc_ids
    except AttributeError:
        pass

    print('Starting inference...')
    i, wav_streams, wpos = sess.run(wave_op, feed_dict=feed_dict)

    print('Writing wav files.')
    for i in range(batch_sz):
        path = '{}/gen.i{}.wav'.format(wav_dir, i)
        librosa.output.write_wav(path, wav_streams[i], sample_rate)
        print('Wrote {}'.format(path))
    print('Finished.')



if __name__ == '__main__':
    main()

