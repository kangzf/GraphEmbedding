def check_flags(FLAGS):
    assert (FLAGS.sample_num is None or FLAGS.sample_num > 0)
    assert (FLAGS.yeta >= 0)
    # TODO: finish.
