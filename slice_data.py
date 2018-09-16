# parse the rdb files and produce new files of a particular start position and size
# adjust size and start position to the nearest hop
# skip files that can't be sliced to that size (too short)

def create_slice(in_wav, in_mel, out_wav, out_mel, hop_sz, beg, sz):
    import numpy as np
    from sys import stderr
    wav = np.load(in_wav)
    if len(wav) < beg + sz:
        print('Skipping {} of length {}'.format(in_wav, len(wav)), file=stderr)
        return False
    np.save(out_wav, wav[beg:beg + sz])
    wav_len = len(wav)

    mel = np.load(in_mel)
    assert len(mel) * hop_sz == wav_len
    mbeg = beg // hop_sz
    m_sz = sz // hop_sz
    np.save(out_mel, mel[mbeg: mbeg + m_sz])
    return True


def adjust_coords(beg, sz, hop):
    return beg - (beg % hop), sz - (sz % hop)

def parse_rdb(rdb_file):
    samples = []
    with open(rdb_file) as rdb_fh:
        for s in rdb_fh.readlines():
            (vid, wav_path, mel_path) = s.strip().split('\t')
            samples.append([int(vid), wav_path, mel_path])
    return samples

def out_path(in_path, sub_dir, out_dir):
    from os import path as osp 
    return '{}/{}/{}'.format(out_dir, sub_dir, 
            osp.basename(in_path).replace('.npy', '.slice.npy'))

def maybe_mkdir(path):
    import os
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Slice Data')
    parser.add_argument('--hop-size', '-hs', type=int, default=256, metavar='INT',
            help='Hop size of the Mel files')
    parser.add_argument('--start-pos', '-sp', type=int, default=1024, metavar='INT',
            help='Start position for the slice')
    parser.add_argument('--slice-size', '-ss', type=int, default=20480, metavar='INT',
            help='Size of the slice')

    # positional arguments
    parser.add_argument('in_rdb_file', metavar='RDB_FILE', type=str,
            help='File containing lines:\n'
            '<id1>\t/path/to/sample1.wav.npy\t/path/to/sample1.mel.npy\n'
            '<id2>\t/path/to/sample2.wav.npy\t/path/to/sample2.mel.npy\n')
    parser.add_argument('out_dir', metavar='OUT_DIR', type=str,
            help='Output directory for writing sliced files')
    parser.add_argument('out_rdb_file', metavar='OUT_RDB_FILE', type=str,
            help='Name of the output rdb file')
    return parser.parse_args()

def main():
    args = get_args()
    import numpy as np
    import os
    samples = parse_rdb(args.in_rdb_file)
    beg, sz = adjust_coords(args.start_pos, args.slice_size, args.hop_size)

    # create output directories
    maybe_mkdir(args.out_dir)
    maybe_mkdir(args.out_dir + '/audio')
    maybe_mkdir(args.out_dir + '/mel')

    with open(args.out_rdb_file, 'w') as rfh:
        for vid, wav_path, mel_path in samples:
            out_wav_path = out_path(wav_path, 'audio', args.out_dir)
            out_mel_path = out_path(mel_path, 'mel', args.out_dir)
            if create_slice(wav_path, mel_path, out_wav_path, out_mel_path, args.hop_size,
                    beg, sz):
                print('{}\t{}\t{}'.format(vid, out_wav_path, out_mel_path), file=rfh)


if __name__ == '__main__':
    main()







        
 
