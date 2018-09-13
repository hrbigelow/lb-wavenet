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
    parser.add_argument('--save-interval', '-si', type=int, default=1000, metavar='INT',
            help='Save a checkpoint after this many steps each time')
    parser.add_argument('--tf-debug', '-tdb', action='store_true', default=False,
            help='Enable tf_debug debugging console')
    parser.add_argument('--tf-eager', '-te', action='store_true', default=False,
            help='Enable tf Eager mode')

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

    if args.tf_eager and args.tf_debug:
        print('Error: --tf-debug and --tf-eager cannot both be set', file=stderr)
        exit(1)

    if args.tf_eager:
        tf.enable_eager_execution()

    with open(args.arch_file, 'r') as fp:
        arch = json.load(fp)

    with open(args.par_file, 'r') as fp:
        par = json.load(fp)

    if args.tf_eager:
        sess = None
    else:
        config = tf.ConfigProto()
        #config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        print('Created tf.Session.', file=stderr)

    data_args = {
            'sam_file': args.sam_file,
            'sample_rate': par['sample_rate'],
            'slice_sz': par['slice_sz'],
            'prefetch_sz': par['prefetch_sz'],
            'mel_spectrum_sz': arch['n_lc_in'],
            'mel_hop_sz': arch['lc_hop_sz'],
            'batch_sz': par['batch_sz']
            }
    dset = data.MaskedSliceWav(sess=sess, **data_args)
    dset.init_sample_catalog()
    n_gc_category = dset.get_max_id()

    # tfdbg can't run if this is before dset.wav_dataset call
    if args.tf_debug: sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    net = tmodel.WaveNetTrain(sess,
            add_summary=args.add_summary, 
            n_gc_category=n_gc_category, **arch, **par)
    dset.set_receptive_field_size(net.get_recep_field_sz())
    wav_dset = dset.wav_dataset()

    dev_string = '/cpu:0' if args.cpu_only else '/gpu:0'

    with contextlib.ExitStack() as stack:
        if args.prof_dir is not None:
            ctx = tf.contrib.tfprof.ProfileContext(args.prof_dir)
            ctx_obj = stack.enter_context(ctx)
        stack.enter_context(tf.device(dev_string))
            
        # create the ops just once if not in eager mode
        if not args.tf_eager:
            wav_input_op, mel_input_op, id_mask_op = dset.wav_dataset_ops(wav_dset) 
            print('Created dataset.', file=stderr)
            grads_and_vars_op, loss_op = \
                    net.grad_var_loss(wav_input_op, mel_input_op, id_mask_op)
            print('Built graph.', file=stderr)

            sess.run(tf.global_variables_initializer())
            print('Initialized training graph.', file=stderr)
            
            apply_grads_op = optimizer.apply_gradients(grads_and_vars)
            print('Created gradients.', file=stderr)

        else:
            # must call this to create the variables
            itr = dset.wav_dataset_itr(wav_dset)
            wav_input, mel_input, id_mask = next(itr) 
            _ = net.build_graph(wav_input, mel_input, id_mask)
            assert len(net.trainable_vars) > 0
            # it seems eager variables are initialized on creation
            #tf.global_variables_initializer()

        if args.resume_step: 
            ckpt = '{}-{}'.format(args.ckpt_path, args.resume_step)
            print('Restoring from {}'.format(ckpt))
            net.restore(sess, ckpt) 


        summary_op = tf.summary.merge_all() if args.add_summary else None
        if summary_op is not None and args.tb_dir is None:
            print('Error: must provide --tb-dir argument if '
            + 'there are summaries in the graph', file=stderr)
            exit(1)

        if args.tb_dir:
            fw = tf.summary.FileWriter(tb_dir, graph=sess.graph)
            make_flusher(fw)

        optimizer = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])
        print('Created optimizer.', file=stderr)

        print('Starting training...', file=stderr)
        step = args.resume_step or 1
        wav_itr = dset.wav_dataset_itr(wav_dset)
        while step < max_steps:
            if args.tf_eager:
                wav_input, mel_input, id_mask = next(wav_itr) 
                grads_and_vars, loss = net.grad_var_loss_eager(wav_input, mel_input, id_mask)
                optimizer.apply_gradients(grads_and_vars)

            else:
                if step == 5:
                    run_meta = tf.RunMetadata()
                    run_opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                    output_partition_graphs=True)
                    _, _ = sess.run([apply_grads, loss],
                            options=run_opts, run_metadata=run_meta)
                    with open('/tmp/run.txt', 'w') as out:
                        out.write(str(run_meta))
                    
                    if args.timeline_file is not None:
                        fetched_timeline = timeline.Timeline(run_meta.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open(args.timeline_file, 'w') as f:
                            f.write(chrome_trace)

                _, loss = sess.run([apply_grads_op, loss_op])

            if step % 10 == 0:
                print('step, loss: {}\t{}'.format(step, loss), file=stderr)
                if args.tf_eager and summary_op is not None:
                    fw.add_summary(sess.run(summary_op), step)

            if step % args.save_interval == 0:
                path = net.save(args.ckpt_path, step)
                print('Saved checkpoint to %s\n' % path, file=stderr)

            step += 1


if __name__ == '__main__':
    main()

