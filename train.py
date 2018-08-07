# training functions

base = '/home/hrbigelow/ai/'
tb_dir = base + 'tb/lb-wavenet'
max_steps = 300000 

def make_flusher(file_writer):
    import signal
    def cleanup(sig, frame):
        print('Flushing file_writer...', end='')
        file_writer.flush()
        print('done.')
        exit(1)
    signal.signal(signal.Signals.SIGINT, cleanup) 

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='WaveNet')
    parser.add_argument('--timeline-file', type=str,
            help='Enable profiling and write info to <timeline_file>')
    parser.add_argument('--prof-dir', type=str, metavar='DIR',
            help='Output profiling events to <prof_dir> for use with ' +
            'TensorFlow Profiler')
    parser.add_argument('--ckpt-pfx', type=str, metavar='STR',
            help='Prefix for loading or storing checkpoint files')
    parser.add_argument('--resume-step', '-r', type=int, metavar='INT',
            help='Resume training from '
            + 'CKPT_DIR/<ckpt_pfx>-<resume_step>.{meta,index,data-..}')
    parser.add_argument('--add-summary', action='store_true',
            help='If present, add summary histogram nodes to graph for TensorBoard')

    # positional arguments
    parser.add_argument('ckpt_dir', type=str, metavar='CKPT_DIR',
            help='Directory for all checkpoints')
    parser.add_argument('arch_file', type=str, metavar='ARCH_FILE',
            help='JSON file specifying architectural parameters')
    parser.add_argument('par_file', type=str, metavar='PAR_FILE',
            help='JSON file specifying training and other hyperparameters')
    parser.add_argument('sam_file', type=str, metavar='SAMPLES_FILE',
            help='File containing lines:\n'
            + '<id>\t/path/to/sample1.wav\n'
            + '<id2>\t/path/to/sample2.wav\n')

    return parser.parse_args()


def main():
    args = get_args()

    import tmodel
    import data
    import tensorflow as tf
    import json
    import contextlib
    from tensorflow.python import debug as tf_debug
    from tensorflow.python.client import timeline
    import tests
    from sys import stderr
    from os.path import join as path_join

    with open(args.arch_file, 'r') as fp:
        arch = json.load(fp)

    with open(args.par_file, 'r') as fp:
        par = json.load(fp)

    ckpt_path = path_join(args.ckpt_dir, args.ckpt_pfx) 

    net = tmodel.WaveNetTrain(
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
            par['l2_factor'],
            par['batch_sz'],
            args.add_summary
            )
    recep_field_sz = net.get_recep_field_sz()


    dset = data.MaskedSliceWav(
            args.sam_file,
            par['batch_sz'],
            par['sample_rate'],
            par['slice_sz'],
            par['prefetch_sz'],
            recep_field_sz
            )

    dset.init_sample_catalog()

    with contextlib.ExitStack() as stack:
        if args.prof_dir is not None:
            ctx = tf.contrib.tfprof.ProfileContext(args.prof_dir)
            ctx_obj = stack.enter_context(ctx)

        sess = tf.Session()
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        wav_input, id_masks, id_maps = dset.wav_dataset(sess)
        print('Created dataset.', file=stderr)

        # Note that tfdbg can't run if this is before dset.wav_dataset call
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        loss = net.build_graph(wav_input, id_masks, id_maps)

        if args.resume_step: 
            ckpt = '{}-{}'.format(ckpt_path, args.resume_step)
            print('Restoring from {}'.format(ckpt))
            net.restore(sess, ckpt) 
        # print(sess.run(wav_input))

        summary_op = tf.summary.merge_all()

        fw = tf.summary.FileWriter(tb_dir, graph=sess.graph)
        make_flusher(fw)
        print('Created training graph.', file=stderr)

        optimizer = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])
        train_vars = tf.trainable_variables()

        # writing this out explicitly for educational purposes
        step = args.resume_step or 0
        global_step = tf.Variable(step, trainable=False)
        grads_and_vars = optimizer.compute_gradients(loss, train_vars)
        apply_grads = optimizer.apply_gradients(grads_and_vars, global_step)


        init = tf.global_variables_initializer()
        sess.run(init)
        print('Initialized training graph.', file=stderr)
        
        #ts = tests.get_tensor_sizes(sess)
        #for t in ts:
        #    print('{}\t{}\t{}\t{}'.format(t[0], t[1], t[2], t[3].size))
        #exit(1)

        sess.run(global_step.initializer)

        print('Starting training...', file=stderr)
           
        while step < max_steps:
            if step == 5 and args.timeline_file is not None:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                _, step, _ = sess.run([apply_grads, global_step, loss],
                        options=options,
                        run_metadata=run_metadata)
                
                fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(args.timeline_file, 'w') as f:
                    f.write(chrome_trace)

            else:
                _, step, loss_val = sess.run([apply_grads, global_step, loss])
            if step % 10 == 0:
                print('step, loss: {}\t{}'.format(step, loss_val), file=stderr)
                if summary_op is not None:
                    fw.add_summary(sess.run(summary_op), step)
            if step % 100 == 0:
                path = net.save(sess, ckpt_path, step)
                print('Saved checkpoint to %s\n' % path, file=stderr)


if __name__ == '__main__':
    main()

