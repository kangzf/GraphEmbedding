from __future__ import division
from __future__ import print_function

from config import FLAGS, placeholders
from train import train_val, test
from utils_siamese import check_flags
from data_siamese import SiameseModelData
from dist_calculator import DistCalculator
from models import GCNTN
from saver import Saver
import tensorflow as tf


def main(FLAGS, placeholders):
    check_flags(FLAGS)
    data = SiameseModelData(FLAGS)
    dist_calculator = DistCalculator(
        FLAGS.dataset, FLAGS.dist_metric, FLAGS.dist_algo)
    if FLAGS.model == 'siamese_gcntn':
        model_func = GCNTN
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    model = model_func(
        FLAGS, placeholders, input_dim=data.input_dim(), logging=FLAGS.log)
    sess = tf.Session()
    saver = Saver(FLAGS, sess)
    sess.run(tf.global_variables_initializer())
    train_costs, train_times, val_costs, val_times = \
        train_val(
            FLAGS, data, placeholders, dist_calculator, model, saver, sess)
    results = \
        test(FLAGS, data, placeholders, dist_calculator, model, saver, sess)
    return train_costs, train_times, val_costs, val_times, results


if __name__ == '__main__':
    main(FLAGS, placeholders)
