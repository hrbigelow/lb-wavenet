# training functions

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
    parser.add_argument('--timeline-file', '-tf', type=str,
            help='Enable profiling and write info to <timeline_file>')
    parser.add_argument('--prof-dir', '-pd', type=str, metavar='DIR',
            help='Output profiling events to <prof_dir> for use with ' +
            'TensorFlow Profiler')
    parser.add_argument('--resume-step', '-r', type=int, metavar='INT',
            help='Resume training from '
            + 'CKPT_DIR/<ckpt_pfx>-<resume_step>.{meta,index,data-..}')
    parser.add_argument('--add-summary', '-s', action='store_true', default=False,
            help='If present, add summary histogram nodes to graph for TensorBoard')
    parser.add_argument('--cpu-only', '-cpu', action='store_true', default=False,
            help='If present, do all computation on CPU')
    parser.add_argument('--tb-dir', '-tb', type=str, metavar='DIR',
            help='TensorBoard directory for writing summary events')

    # positional arguments
    parser.add_argument('ckpt_path', type=str, metavar='CKPT_PATH_PFX',
            help='E.g. /path/to/ckpt/pfx, a path and prefix combination for writing checkpoint files')
    parser.add_argument('arch_file', type=str, metavar='ARCH_FILE',
            help='JSON file specifying architectural parameters')
    parser.add_argument('par_file', type=str, metavar='PAR_FILE',
            help='JSON file specifying training and other hyperparameters')
    parser.add_argument('sam_file', type=str, metavar='SAMPLES_FILE',
            help='File containing lines:\n'
            + '<id1>\t/path/to/sample1.wav.npy\t/path/to/sample1.mel.npy\n'
            + '<id2>\t/path/to/sample2.wav.npy\t/path/to/sample2.mel.npy\n')

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


    data_args = {
            'sam_file': args.sam_file,
            'sample_rate': par['sample_rate'],
            'slice_sz': par['slice_sz'],
            'prefetch_sz': par['prefetch_sz'],
            'mel_spectrum_sz': arch['n_lc_in'],
            'mel_hop_sz': arch['lc_hop_sz'],
            'batch_sz': par['batch_sz']
            }
    dset = data.MaskedSliceWav(**data_args)
    dset.init_sample_catalog()
    n_gc_category = dset.get_max_id()

    net_args = { 'add_summary': args.add_summary,
            'n_gc_category': n_gc_category, **arch, **par }
    net = tmodel.WaveNetTrain(**net_args)

    dset.set_receptive_field_size(net.get_recep_field_sz())

    dev_string = '/cpu:0' if args.cpu_only else '/gpu:0'

    with contextlib.ExitStack() as stack:
        if args.prof_dir is not None:
            ctx = tf.contrib.tfprof.ProfileContext(args.prof_dir)
            ctx_obj = stack.enter_context(ctx)
            
        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        print('Created tf.Session.', file=stderr)

        with tf.device(dev_string):
            wav_input, mel_input, id_masks = dset.wav_dataset(sess)
            print('Created dataset.', file=stderr)

            # Note that tfdbg can't run if this is before dset.wav_dataset call
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            loss = net.build_graph(wav_input, mel_input, id_masks)
            print('Built graph.', file=stderr)

        if args.resume_step: 
            ckpt = '{}-{}'.format(args.ckpt_path, args.resume_step)
            print('Restoring from {}'.format(ckpt))
            net.restore(sess, ckpt) 
        # print(sess.run(wav_input))

        summary_op = tf.summary.merge_all()

        if summary_op is not None and args.tb_dir is None:
            print('Error: must provide --tb-dir argument if '
            + 'there are summaries in the graph', file=stderr)
            exit(1)

        if args.tb_dir:
            fw = tf.summary.FileWriter(tb_dir, graph=sess.graph)
            make_flusher(fw)
        print('Created training graph.', file=stderr)

        with tf.device(dev_string):
            optimizer = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])
            train_vars = tf.trainable_variables()
            print('Created optimizer.', file=stderr)

            # writing this out explicitly for educational purposes
            step = args.resume_step or 0
            global_step = tf.Variable(step, trainable=False)
            grads_and_vars = optimizer.compute_gradients(loss, train_vars)
            apply_grads = optimizer.apply_gradients(grads_and_vars, global_step)
            print('Created gradients.', file=stderr)

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
            if step == 5:
                run_meta = tf.RunMetadata()
                run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                output_partition_graphs=True)
                _, step, _ = sess.run([apply_grads, global_step, loss],
                        options=run_opts, run_metadata=run_meta)
                with open('/tmp/run.txt', 'w') as out:
                    out.write(str(run_meta))
                
                if args.timeline_file is not None:
                    fetched_timeline = timeline.Timeline(run_meta.step_stats)
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
                path = net.save(sess, args.ckpt_path, step)
                print('Saved checkpoint to %s\n' % path, file=stderr)


if __name__ == '__main__':
    main()

