from config import FLAGS
import sys
from os.path import dirname, abspath

cur_folder = dirname(abspath(__file__))
sys.path.insert(0, '{}/../../src'.format(cur_folder))


def solve_parent_dir():
    pass


def check_flags():
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


def get_model_info_as_str(model_info_table=None):
    rtn = ''
    for k, v in sorted(FLAGS.flag_values_dict().items(), key=lambda x: x[0]):
        s = '{0:26} : {1}\n'.format(k, v)
        rtn += s
        if model_info_table:
            model_info_table.append([k, '**{}**'.format(v)])
    return rtn
