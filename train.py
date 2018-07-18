# training functions
import tmodel
import data
import tensorflow as tf
import json

sam_file = 'samples.rdb'
log_dir = '/home/hrbigelow/ai/ckpt'
par_dir = '/home/hrbigelow/ai/par'
arch_file = 'arch1.json'
par_file = 'par1.json'
max_steps = 1000

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
            par['l2_factor']
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
    (wav_input, id_masks, id_maps) = dset.wav_dataset(sess)
    print('Created dataset.')

    loss = net.build_graph(wav_input, id_masks, id_maps)
    print('Created training graph.')

    optimizer = tf.train.AdamOptimizer()
    train_vars = tf.trainable_variables()

    # writing this out explicitly for educational purposes
    global_step = tf.Variable(0, trainable=False)
    grads_and_vars = optimizer.compute_gradients(loss, train_vars)
    apply_grads = optimizer.apply_gradients(grads_and_vars, global_step)


    init = tf.global_variables_initializer()
    sess.run(init)
    print('Initialized training graph.')
    #input('Continue?')

    sess.run(global_step.initializer)

    #input('Continue?')
    print('Starting training')
    while step < max_steps:
        _, step, loss_val = sess.run([apply_grads, global_step, loss])
        print('step, loss: {}\t{}'.format(step, loss_val))
        if step % 100 == 0:
            net.save(sess, log_dir, arch_file, step)
        


if __name__ == '__main__':
    main()


