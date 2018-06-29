# training functions
import model
import data
import tensorflow as tf

def main():

    n_blocks = 3
    n_block_layers = 10
    n_quant_chan = 256
    n_res_chan = 32
    n_dil_chan = 16
    n_skip_chan = 32
    n_post1_chan = 100
    n_gc_embed_chan = 100
    n_gc_category = 450
    l2_factor = 0.1
    sam_path = 'samples.rdb'
    slice_sz = 1000
    batch_sz = 16
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
    (wav_input, id_mask, id_map) = dset.wav_dataset(sess)

    loss = net.create_training_graph(wav_input, id_mask, id_map)
    optimizer = tf.train.AdamOptimizer()
    train_vars = tf.trainable_variables()

    # writing this out explicitly for educational purposes
    global_step = tf.Variable(0)
    grads_and_vars = optimizer.compute_gradients(loss, train_vars)
    apply_grads = optimizer.apply_gradients(grads_and_vars, global_step)

    net.initialize_training_graph(sess) 

    while True:
        sess.run(apply_grads)


if __name__ == '__main__':
    main()






