# training functions
import model as wn 
import data as ds
import tensorflow as tf

def main():

    (n_blocks, n_block_layers, n_in_chan, n_res_chan, n_dil_chan,
            n_skip_chan, n_post1_chan, n_quant_chan, n_gc_embed_chan, n_gc_category, l2_factor) = ()
    (sam_path, slice_sz, batch_sz, sample_rate) = ()

    net = wn.WaveNet(n_blocks, n_block_layers, n_in_chan, n_res_chan, n_dil_chan,
            n_skip_chan, n_post1_chan, n_quant_chan, n_gc_embed_chan, l2_factor)
    recep_field_sz = net.get_recep_field_sz()

    sess = tf.Session()
    (wav_input, id_mask, id_map) = ds.wav_dataset(
            sam_path, slice_sz, batch_sz, recep_field_sz, sample_rate, sess)

    loss = net.create_training_graph(wav_input, id_mask, id_map)
    optimizer = tf.train.AdamOptimizer()
    train_vars = tf.trainable_variables()






