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
    parser.add_argument('--resume-step', '-rs', type=int, metavar='INT',
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
    parser.add_argument('--progress-interval', '-pi', type=int, default=10, metavar='INT',
            help='Print a progress message at this interval')
    parser.add_argument('--tf-debug', '-tdb', action='store_true', default=False,
            help='Enable tf_debug debugging console')
    parser.add_argument('--tf-eager', '-te', action='store_true', default=False,
            help='Enable tf Eager mode')

    # Training parameter overrides
    parser.add_argument('--batch-size', '-bs', type=int, metavar='INT',
            help='Batch size (overrides PAR_FILE setting)')
    parser.add_argument('--slice-size', '-ss', type=int, metavar='INT',
            help='Slice size (overrides PAR_FILE setting)')
    parser.add_argument('--l2-factor', '-l2', type=float, metavar='FLOAT',
            help='Loss = Xent loss + l2_factor * l2_loss')
    parser.add_argument('--learning-rate', '-lr', type=float, metavar='FLOAT',
            help='Learning rate (overrides PAR_FILE setting)')
    parser.add_argument('--num-global-cond', '-gc', type=int, metavar='INT',
            help='Number of global conditioning categories')

    # positional arguments
    parser.add_argument('ckpt_path', type=str, metavar='CKPT_PATH_PFX',
            help='E.g. /path/to/ckpt/pfx, a path and '
            'prefix combination for writing checkpoint files')
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

    import json
    from sys import stderr

    with open(args.arch_file, 'r') as fp:
        arch = json.load(fp)

    with open(args.par_file, 'r') as fp:
        par = json.load(fp)

    # args consistency checks
    if args.num_global_cond is None and 'n_gc_category' not in arch:
        print('Error: must provide n_gc_category in ARCH_FILE, or --num-global-cond',
                file=stderr)
        exit(1)

    if args.tf_eager and args.tf_debug:
        print('Error: --tf-debug and --tf-eager cannot both be set', file=stderr)
        exit(1)

    import tmodel
    import data
    import tensorflow as tf
    import contextlib
    from tensorflow.python import debug as tf_debug
    from tensorflow.python.client import timeline
    import tests
    from os.path import join as path_join


    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.2
    # config.log_device_placement = True
    #config.gpu_options.allow_growth = True
    #config.allow_soft_placement = True

    if args.tf_eager:
        tf.enable_eager_execution(config=config)

    # Overrides
    if args.batch_size is not None:
        par['batch_sz'] = args.batch_size

    if args.slice_size is not None:
        par['slice_sz'] = args.slice_size

    if args.l2_factor is not None:
        par['l2_factor'] = args.l2_factor
        
    if args.learning_rate is not None:
        par['learning_rate'] = args.learning_rate


    if args.tf_eager:
        sess = None
    else:
        sess = tf.Session(config=config)
        print('Created tf.Session.', file=stderr)

    from functools import reduce
    mel_hop_sz = reduce(lambda x, y: x * y, arch['lc_upsample']) 
    
    dset = data.MaskedSliceWav(sess, args.sam_file, par['sample_rate'],
            par['slice_sz'], par['prefetch_sz'], arch['n_lc_in'],
            mel_hop_sz, par['batch_sz'])
            
    dset.init_sample_catalog()
        
    if args.num_global_cond is not None:
        if args.num_global_cond < dset.get_max_id():
            print('Error: --num-global-cond must be >= {}, the highest ID in the dataset.'.format(
                dset.get_max_id()), file=stderr)
            exit(1)
        else:
            arch['n_gc_category'] = args.num_global_cond

    # tfdbg can't run if this is before dset.wav_dataset call
    if args.tf_debug: sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    net = tmodel.WaveNetTrain(**arch,
            batch_sz = par['batch_sz'],
            l2_factor=par['l2_factor'],
            add_summary=par['add_summary'],
            n_keep_checkpoints=par['n_keep_checkpoints'],
            sess=sess,
            print_interval=args.progress_interval,
            initial_step=args.resume_step or 0
            )

    dset.set_receptive_field_size(net.get_recep_field_sz())
    wav_dset = dset.wav_dataset()

    dev_string = '/cpu:0' if args.cpu_only else '/gpu:0'

    with contextlib.ExitStack() as stack:
        if args.prof_dir is not None:
            ctx = tf.contrib.tfprof.ProfileContext(args.prof_dir)
            ctx_obj = stack.enter_context(ctx)
        #stack.enter_context(tf.device(dev_string))
            
        optimizer = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])

        # create the ops just once if not in eager mode
        if not args.tf_eager:
            wav_input_op, mel_input_op, id_mask_op = dset.wav_dataset_ops(wav_dset) 
            print('Created dataset.', file=stderr)
            grads_and_vars_op, loss_op = \
                    net.grad_var_loss(wav_input_op, mel_input_op, id_mask_op)
            print('Built graph.', file=stderr)

            apply_grads_op = optimizer.apply_gradients(grads_and_vars_op)
            print('Created gradients.', file=stderr)

        else:
            # must call this to create the variables
            itr = dset.wav_dataset_itr(wav_dset)
            wav_input, mel_input, id_mask = next(itr) 
            _ = net.build_graph(wav_input, mel_input, id_mask)
            assert len(net.vars) > 0

        net.init_vars()
        print('Initialized training graph.', file=stderr)

        if args.resume_step: 
            ckpt = '{}-{}'.format(args.ckpt_path, args.resume_step)
            print('Restoring from {}'.format(ckpt))
            net.restore(ckpt) 


        summary_op = tf.summary.merge_all() if args.add_summary else None
        if summary_op is not None and args.tb_dir is None:
            print('Error: must provide --tb-dir argument if '
            + 'there are summaries in the graph', file=stderr)
            exit(1)

        if args.tb_dir:
            fw = tf.summary.FileWriter(tb_dir, graph=sess.graph)
            make_flusher(fw)

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
                    _, _ = sess.run([apply_grads_op, loss_op],
                            options=run_opts, run_metadata=run_meta)
                    with open('/tmp/run.txt', 'w') as out:
                        out.write(str(run_meta))
                    
                    if args.timeline_file is not None:
                        fetched_timeline = timeline.Timeline(run_meta.step_stats)
                        chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        with open(args.timeline_file, 'w') as f:
                            f.write(chrome_trace)

                _, loss = sess.run([apply_grads_op, loss_op])

            if step % 1 == 0:
                if args.tf_eager and summary_op is not None:
                    fw.add_summary(sess.run(summary_op), step)

            if step % args.save_interval == 0 and step != args.resume_step:
                path = net.save(args.ckpt_path, step)
                print('Saved checkpoint to %s\n' % path, file=stderr)

            step += 1


if __name__ == '__main__':
    main()

