# training functions
import model
import data
import tensorflow as tf

def main():

    n_blocks = 3
    n_block_layers = 10 
    n_quant_chan = 256 
    n_res_chan = 16 
    n_dil_chan = 32 
    n_skip_chan = 16 
    n_post1_chan = 512 
    #n_gc_embed_chan = 0
    n_gc_embed_chan = 17 
    n_gc_category = 377 
    l2_factor = 0.1
    sam_path = 'samples.rdb'
    slice_sz = 100
    batch_sz = 5 
    sample_rate = 16000

    net = model.WaveNet(
            n_blocks, n_block_layers, n_quant_chan,
            n_res_chan, n_dil_chan, n_skip_chan,
            n_post1_chan, n_gc_embed_chan, n_gc_category,
            l2_factor
            )
    recep_field_sz = net.get_recep_field_sz()

    dset = data.MaskedSliceWav(sam_path, batch_sz, sample_rate, slice_sz, recep_field_sz)
    dset.init_sample_catalog()

    sess = tf.Session()
    (wav_input, id_masks, id_maps) = dset.wav_dataset(sess)
    print('Created dataset.')

    loss = net.create_training_graph(wav_input, id_masks, id_maps)
    print('Created training graph.')

    optimizer = tf.train.AdamOptimizer()
    train_vars = tf.trainable_variables()

    # writing this out explicitly for educational purposes
    global_step = tf.Variable(0)
    grads_and_vars = optimizer.compute_gradients(loss, train_vars)
    apply_grads = optimizer.apply_gradients(grads_and_vars, global_step)


    init = tf.global_variables_initializer()
    sess.run(init)

    net.initialize_training_graph(sess) 
    print('Initialized training graph.')
    sess.run(global_step.initializer)

    print('Starting training')
    while True:
        _, step, loss_val = sess.run([apply_grads, global_step, loss])
        print('step, loss: {}\t{}'.format(step, loss_val))
        


if __name__ == '__main__':
    main()






