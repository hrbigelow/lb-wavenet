# training functions
import tmodel
import data
import tensorflow as tf
import json
import signal
from tensorflow.python import debug as tf_debug


sam_file = '/home/hrbigelow/ai/data/vctk_samples.rdb'
ckpt_dir = '/home/hrbigelow/ai/ckpt/lb-wavenet'
tb_dir = '/home/hrbigelow/ai/tb/lb-wavenet'
par_dir = '/home/hrbigelow/ai/par'
arch_file = 'arch3.json'
par_file = 'par1.json'
max_steps = 30000 
add_summary = True

def make_flusher(file_writer):
    def cleanup(sig, frame):
        print('Flushing file_writer...', end='')
        file_writer.flush()
        print('done.')
        exit(1)
    signal.signal(signal.Signals.SIGINT, cleanup) 


def main():

    with open(par_dir + '/' + arch_file, 'r') as fp:
        arch = json.load(fp)

    with open(par_dir + '/' + par_file, 'r') as fp:
        par = json.load(fp)

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
            add_summary
            )
    recep_field_sz = net.get_recep_field_sz()

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
    print('Created dataset.')

    # Note that tfdbg can't run if this is before dset.wav_dataset call
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    loss = net.build_graph(wav_input, id_masks, id_maps)
    # print(sess.run(wav_input))

    summary_op = tf.summary.merge_all()
    fw = tf.summary.FileWriter(tb_dir, graph=sess.graph)
    make_flusher(fw)
    print('Created training graph.')

    optimizer = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])
    train_vars = tf.trainable_variables()

    # writing this out explicitly for educational purposes
    global_step = tf.Variable(0, trainable=False)
    grads_and_vars = optimizer.compute_gradients(loss, train_vars)
    apply_grads = optimizer.apply_gradients(grads_and_vars, global_step)

    init = tf.global_variables_initializer()
    sess.run(init)
    print('Initialized training graph.')

    sess.run(global_step.initializer)

    print('Starting training...')
    step = 0
    while step < max_steps:
        _, step, loss_val = sess.run([apply_grads, global_step, loss])
        if step % 10 == 0:
            print('step, loss: {}\t{}'.format(step, loss_val))
            fw.add_summary(sess.run(summary_op), step)
        if step % 100 == 0:
            path = net.save(sess, ckpt_dir, arch_file, step)
            print('Saved checkpoint to %s\n' % path)

        

if __name__ == '__main__':
    main()

