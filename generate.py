# generate audio
import imodel
import tensorflow as tf
import librosa 
import json
import argparse

from tensorflow.python import debug as tf_debug

base = '/home/hrbigelow/ai/'
log_dir = base + 'ckpt/lb-wavenet'
tb_dir = base + 'tb/lb-wavenet'
par_dir = base + 'par'
wav_dir = base + 'wav'


def get_args():
    parser = argparse.ArgumentParser(description='WaveNet')
    parser.add_argument('--arch-file', type=str,
            help='JSON file specifying architectural parameters')
    parser.add_argument('ckpt_pfx', metavar='CHECKPOINT_PREFIX', type=str,
            help='Provide <pfx> for <pfx>.{meta,index,data-..} files')
    parser.add_argument('--teacher-wav', type=str,
            help='Provide a preliminary teacher-forcing vector to prime the generation')
    parser.add_argument('--teacher-duration', type=float,
            help='Number of seconds to parse from <teacher_wav>')
    parser.add_argument('--gen-seconds', type=float, default=5,
            help='Number of additional seconds to generate')
    parser.add_argument('--sample-rate', type=int, default=16000,
            help='Number of samples per second for parsed .wav files')
    parser.add_argument('--chunk-size', type=int, default=1000,
            help='Number of timesteps to generate between internal buffer shifts')
    parser.add_argument('--batch-size', type=int, default=10,
            help='Number of .wav files to generate simultaneously')

    return parser.parse_args()


def main():
    args = get_args()

    with open(par_dir + '/' + args.arch_file, 'r') as fp:
        arch = json.load(fp)

    if args.teacher_wav is not None:
        import librosa
        teacher_vec, _ = librosa.load(args.teacher_wav,
                args.sample_rate, duration=args.teacher_duration, mono=True)
        teacher_seconds = teacher_vec.shape[0] / args.sample_rate
    else:
        teacher_seconds = 0

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
            args.batch_sz,
            args.chunk_sz,
            teacher_vec)

    print('Building graph.')
    wave_ops = net.build_graph()

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess = tf.Session()
    ckpt_path = '{}/{}'.format(log_dir, args.ckpt_pfx)
    print('Restoring from {}'.format(ckpt_path))
    net.restore(sess, ckpt_path) 
    #tf.get_default_graph().finalize()

    print('Creating FileWriter.')
    fw = tf.summary.FileWriter(tb_dir, graph=sess.graph)

    print('Initializing buffers.')
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'asus:6064')
    net.init_buffers(sess)

    gc_ids = [5,6]
    gen_sz = (args.gen_seconds + teacher_seconds) * sample_rate

    feed_dict = { net.gen_sz: gen_sz }
    try:
        feed_dict[net.gc_ids] = gc_ids
    except AttributeError:
        pass

    print('Starting inference...')
    n, wav_streams, wpos = sess.run(wave_ops, feed_dict=feed_dict)


    print('Writing wav files.')
    for i in range(batch_sz):
        path = '{}/gen.i{}.wav'.format(wav_dir, i)
        librosa.output.write_wav(path, wav_streams[i], args.sample_rate)
        print('Wrote {}'.format(path))
    print('Finished.')



if __name__ == '__main__':
    main()

