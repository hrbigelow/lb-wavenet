# generate audio
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='WaveNet')
    parser.add_argument('--teacher-wav', '-w', type=str,
            help='Provide a preliminary teacher-forcing vector to prime the generation')
    parser.add_argument('--teacher-start', '-ts', type=float,
            help='Number of seconds to parse from <teacher_wav>')
    parser.add_argument('--teacher-duration', '-td', type=float,
            help='Number of seconds to parse from <teacher_wav>')
    parser.add_argument('--gen-seconds', '-g', type=float, default=5,
            help='Number of additional seconds to generate')
    parser.add_argument('--sample-rate', '-s', type=int, default=16000,
            help='Number of samples per second for parsed .wav files')
    parser.add_argument('--chunk-size', '-c', type=int, default=1000,
            help='Number of timesteps to generate between internal buffer shifts')
    parser.add_argument('--batch-size', '-b', type=int, default=10,
            help='Number of .wav files to generate simultaneously')

    # positional arguments
    parser.add_argument('arch_file', type=str, metavar='ARCH_FILE',
            help='JSON file specifying architectural parameters')
    parser.add_argument('ckpt', metavar='CHECKPOINT_PREFIX', type=str,
            help='Provide <ckpt> for <ckpt>.{meta,index,data-..} files')
    parser.add_argument('wav_dir', metavar='OUTPUT_WAV_DIR', type=str,
            help='Output directory for generated .wav files')
    return parser.parse_args()


def main():
    args = get_args()

    import imodel
    from arch import WaveNetArch 
    import tensorflow as tf
    import librosa 
    import json
    from tensorflow.python import debug as tf_debug
    import os
    from sys import stderr

    with open(args.arch_file, 'r') as fp:
        arch = json.load(fp)

    for fn in WaveNetArch.expand_ckpt(args.ckpt):
        if not os.access(fn, os.R_OK):
            print("Couldn't find checkpoint file {}".format(fn), file=stderr)
            exit(1)

    if args.teacher_wav is not None:
        import librosa
        teacher_vec, _ = librosa.load(args.teacher_wav,
                args.sample_rate, offset=args.teacher_start,
                duration=args.teacher_duration, mono=True)
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
            args.batch_size,
            args.chunk_size,
            teacher_vec)

    print('Building graph.')
    #with tf.device('/gpu:0'):
    wave_ops = net.build_graph()

    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess = tf.Session()
    print('Restoring from {}'.format(args.ckpt))
    net.restore(sess, args.ckpt) 
    #tf.get_default_graph().finalize()

    # print('Creating FileWriter.')
    # fw = tf.summary.FileWriter(tb_dir, graph=sess.graph)

    print('Initializing buffers.')
    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, 'asus:6064')
    net.init_buffers(sess)


    gc_ids = [5,6]
    gen_sz = (args.gen_seconds + teacher_seconds) * args.sample_rate

    feed_dict = { net.gen_sz: gen_sz }
    try:
        feed_dict[net.gc_ids] = gc_ids
    except AttributeError:
        pass

#    import tests
#    tsizes = tests.get_tensor_sizes(sess, feed_dict=feed_dict)
#    for n,s,p,d in tsizes:
#        print('{}\t{}\t{}\t{}'.format(n, s, p, d))
#    exit(1)

    print('Starting inference...')
    n, wav_streams, wpos = sess.run(wave_ops, feed_dict=feed_dict)

    print('Writing wav files.')
    for i in range(args.batch_size):
        path = os.path.join(wav_dir, 'gen.i{}.wav'.format(i))
        librosa.output.write_wav(path, wav_streams[i], args.sample_rate)
        print('Wrote {}'.format(path))
    print('Finished.')



if __name__ == '__main__':
    main()

