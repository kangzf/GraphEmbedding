from __future__ import division
from __future__ import print_function

from config import FLAGS
from train import train_val, test
from utils_siamese import check_flags
from data_siamese import SiameseModelData
from dist_calculator import DistCalculator
from models_factory import create_model
from saver import Saver
import tensorflow as tf


def main():
    check_flags(FLAGS)
    data = SiameseModelData()
    dist_calculator = DistCalculator(
        FLAGS.dataset, FLAGS.dist_metric, FLAGS.dist_algo)
    model = create_model(FLAGS.model, data.input_dim())
    sess = tf.Session()
    saver = Saver(FLAGS, sess)
    sess.run(tf.global_variables_initializer())
    train_costs, train_times, val_costs, val_times = \
        train_val(data, dist_calculator, model, saver, sess)
    results = \
        test(data, dist_calculator, model, saver, sess)
    return train_costs, train_times, val_costs, val_times, results


if __name__ == '__main__':
    main()
