def check_flags(FLAGS):
    assert (0 < FLAGS.valid_percentage < 1)
    assert (FLAGS.sample_num >= -1)
    assert (FLAGS.yeta >= 0)
    assert (FLAGS.num_layers >= 2)
    # TODO: finish.


def print_msec(sec):
    return '{:.2f}msec'.format(sec * 1000)
