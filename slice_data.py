# parse the rdb files and produce new files of a particular start position and size
# adjust size and start position to the nearest hop
# skip files that can't be sliced to that size (too short)

def create_slice(in_wav, in_mel, out_wav, out_mel, hop_sz, beg, sz):
    with open(in_wav, 'r') as iw:
        wav = np.load(iw)
        if len(wav) < beg + sz:
            print('Skipping {} of length {}'.format(in_wav, len(wav)), file=stderr)
            return
        np.save(out_wav, wav[beg:beg + sz])
    with open(in_mel, 'r') as im:
        mel = np.load(im)
        mbeg = beg / hop_sz
        m_sz = sz / hop_sz
        np.save(out_mel, mel[mbeg: mbeg + m_sz])
    return

def adjust_coords(beg, sz, hop):
    return beg - (beg % hop), sz - (sz % hop)

def parse_rdb(rdb_file):
    samples = []
    with open(rdb_file) as rdb_fh:
        for s in rdb_fh.readlines():
            (vid, wav_path, mel_path) = s.strip().split('\t')
            samples.append([int(vid), wav_path, mel_path])
    return samples

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Slice Data')
    parser.add_argument('--hop-size', '-h', type=int, default=256, metavar='INT',
            help='Hop size of the Mel files')
    parser.add_argument('--start-pos', '-sp', type=int, default=1024, metavar='INT',
            help='Start position for the slice')
    parser.add_argument('--slice-size', '-ss', type=int, default=20480, metavar='INT',
            help='Size of the slice')

    # positional arguments
    parser.add_argument('rdb_file', metavar='RDB_FILE', type=str,
            help='File containing lines:\n'
            '<id1>\t/path/to/sample1.wav.npy\t/path/to/sample1.mel.npy\n'
            '<id2>\t/path/to/sample2.wav.npy\t/path/to/sample2.mel.npy\n')
    parser.add_argument('out_dir', metavar='RDB _FILE', type=str,
    return parser.parse_args()

def main():
    args = get_args()
    import numpy as np
    samples = parse_rdb(args.rdb_file)







        
 
