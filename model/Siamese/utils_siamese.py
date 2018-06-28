import sys
from os.path import dirname, abspath

cur_folder = dirname(abspath(__file__))
sys.path.insert(0, '{}/../../src'.format(cur_folder))


def check_flags(FLAGS):
    assert (0 < FLAGS.valid_percentage < 1)
    assert (FLAGS.sample_num >= -1)
    assert (FLAGS.yeta >= 0)
    assert (FLAGS.num_layers >= 2)
    assert (FLAGS.batch_size >= 1)
    # TODO: finish.


def print_msec(sec):
    return '{:.2f}msec'.format(sec * 1000)


def get_siamese_dir():
    return cur_folder


def get_phldr(phldr, key, tvt):
    if 'train' in tvt or 'val' in tvt:
        return phldr[key]
    else:
        return phldr['test_{}'.format(key.replace('inputs', 'input'))]
