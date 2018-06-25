# AUTO TUNING PARAMETERS
# Hyperparameters   | Range        | Note
# ------------------------------------------------------------------------------
# norm_dist          T/F            2 norm/no norm
# yeta               0.2/0.01       5 diff yetas
# activation func    5              5 diff  activation layers
# learning_rate      0.06-0.001     6 diff lr
# iteration          500-2000       4 diff iter
# validation ratio   0.2-0.4        2 diff ratio   
# ------------------------------------------------------------------------------
from utils_siamese import get_siamese_dir
from utils import get_ts
from main import main
from config import FLAGS, placeholders
import numpy as np
import tensorflow as tf
import csv
import itertools

dataset = 'aids50nef'
file_name = '{}/logs/parameter_tuning_{}_{}.csv'.format(
    get_siamese_dir(), dataset, get_ts())

header = ['norm_dist', 'yeta', 'final_act', 'learning_rate', 'iter',
          'validation_ratio',
          'best_train_loss', 'best_train_loss_iter', 'best_val_loss',
          'best_val_loss_iter',
          'prec@1_norm', 'prec@2_norm', 'prec@3_norm', 'prec@4_norm',
          'prec@5_norm', 'prec@6_norm', 'prec@7_norm',
          'prec@8_norm', 'prec@9_norm',
          'prec@10_norm', 'prec@1_nonorm', 'prec@2_nonorm', 'prec@3_nonorm',
          'prec@4_nonorm', 'prec@5_nonorm',
          'prec@6_nonorm', 'prec@7_nonorm',
          'prec@8_nonorm', 'prec@9_nonorm', 'prec@10_nonorm', 'mrr_norm',
          'mrr_nonorm', 'mse_norm', 'mse_nonorm']

norm_dist_range = [True, False]
yeta_range_norm = [0.1, 0.2, 0.3, 0.5, 0.8]
yeta_range_nonorm = [0.1, 0.05, 0.01, 0.005, 0.001]
final_act_range = ['identity', 'relu', 'sigmoid', 'tanh', 'sim_kernel']
lr_range = [0.06, 0.03, 0.01, 0.006, 0.003, 0.001]
iter_range = list(range(500, 2001, 500))
val_ratio_range = [0.2, 0.4]


def tune(FLAGS, placeholders):
    f = setup_file()
    best_results_train_loss, best_results_val_loss = float('Inf'), float('Inf')
    results_train = []
    results_val = []
    i = 1
    for norm_dist in norm_dist_range:
        yeta_range = yeta_range_norm if norm_dist == True else yeta_range_nonorm
        for yeta, final_act, lr, iteration, val_ratio in itertools.product(
                yeta_range, final_act_range, lr_range, iter_range,
                val_ratio_range):
            print('Number of tuning iteration: {}'.format(i),
                  'norm_dist: {}'.format(norm_dist), 'yeta: {}'.format(yeta),
                  'final_act: {}'.format(final_act),
                  'learning_rate: {}'.format(lr),
                  'iteration: {}'.format(iteration),
                  'validation_ratio: {}'.format(val_ratio))
            i += 1
            flags = tf.app.flags
            # # reset_flag(FLAGS, flags.DEFINE_string, 'dataset', dataset)
            # reset_flag(FLAGS, flags.DEFINE_float, 'valid_percentage',
            #            val_ratio)
            # reset_flag(FLAGS, flags.DEFINE_bool, 'norm_dist', norm_dist)
            # reset_flag(FLAGS, flags.DEFINE_float, 'yeta', yeta)
            # reset_flag(FLAGS, flags.DEFINE_string, 'final_act', final_act)
            # reset_flag(FLAGS, flags.DEFINE_float, 'learning_rate', lr)
            # reset_flag(FLAGS, flags.DEFINE_integer, 'iters', iteration)
            # reset_flag(FLAGS, flags.DEFINE_bool, 'log', False)
            # reset_flag(FLAGS, flags.DEFINE_bool, 'plot_results', False)
            # FLAGS = tf.app.flags.FLAGS
            train_costs, train_times, val_costs, val_times, results \
                = main(FLAGS, placeholders)
            best_train_loss = np.min(train_costs)
            best_train_iter = np.argmin(train_costs)
            best_val_loss = np.min(val_costs)
            best_val_iter = np.argmin(val_costs)
            print('best_train_loss: {}'.format(best_train_loss),
                  'best_val_loss: {}'.format(best_val_loss),
                  'best_train_iter: {}'.format(best_train_iter),
                  'best_val_iter: {}'.format(best_val_iter))

            model_results = parse_results(results)
            csv_record([[str(x) for x in
                         [norm_dist, yeta, final_act, lr, iteration,
                          val_ratio,
                          best_train_loss, best_train_iter, best_val_loss,
                          best_val_iter] + model_results]], f)

            if best_train_loss < best_results_train_loss:
                best_results_train_loss = best_train_loss
                results_train = [norm_dist, yeta, final_act, lr,
                                 iteration, val_ratio,
                                 best_train_loss, best_train_iter] + \
                                model_results

            if best_val_loss < best_results_val_loss:
                best_results_val_loss = best_val_loss
                results_val = [norm_dist, yeta, final_act, lr,
                               iteration, val_ratio,
                               best_val_loss, best_val_iter] + model_results

    print(results_train)
    print(results_val)

    csv_record([['Final resultss:']], f)
    csv_record([['norm_dist', 'yeta', 'final_act', 'learning_rate', 'iter',
                 'validation_ratio',
                 'best_train_loss', 'best_train_loss_iter',
                 'prec@1_norm', 'prec@2_norm', 'prec@3_norm', 'prec@4_norm',
                 'prec@5_norm', 'prec@6_norm', 'prec@7_norm',
                 'prec@8_norm', 'prec@9_norm',
                 'prec@10_norm', 'prec@1_nonorm', 'prec@2_nonorm',
                 'prec@3_nonorm',
                 'prec@4_nonorm', 'prec@5_nonorm',
                 'prec@6_nonorm', 'prec@7_nonorm',
                 'prec@8_nonorm', 'prec@9_nonorm', 'prec@10_nonorm', 'mrr_norm',
                 'mrr_nonorm', 'mse_norm', 'mse_nonorm']], f)
    csv_record([[str(x) for x in results_train]], f)

    csv_record([['norm_dist', 'yeta', 'final_act', 'learning_rate', 'iter',
                 'validation_ratio',
                 'best_val_loss', 'best_val_loss_iter',
                 'prec@1_norm', 'prec@2_norm', 'prec@3_norm', 'prec@4_norm',
                 'prec@5_norm', 'prec@6_norm', 'prec@7_norm',
                 'prec@8_norm', 'prec@9_norm',
                 'prec@10_norm', 'prec@1_nonorm', 'prec@2_nonorm',
                 'prec@3_nonorm',
                 'prec@4_nonorm', 'prec@5_nonorm',
                 'prec@6_nonorm', 'prec@7_nonorm',
                 'prec@8_nonorm', 'prec@9_nonorm', 'prec@10_nonorm', 'mrr_norm',
                 'mrr_nonorm', 'mse_norm', 'mse_nonorm']], f)
    csv_record([[str(x) for x in results_val]], f)


def setup_file():
    f = open(file_name, 'w')
    writer = csv.writer(f)
    writer.writerows([header])
    return f


def parse_results(results):
    mrr_norm = results['mrr_norm']['siamese_gcntn']
    mrr_nonorm = results['mrr_nonorm']['siamese_gcntn']
    mse_norm = results['mse_norm']['siamese_gcntn']
    mse_nonorm = results['mse_nonorm']['siamese_gcntn']
    apk_norm = results['apk_norm']['siamese_gcntn']['aps'][:10]
    apk_nonorm = results['apk_nonorm']['siamese_gcntn']['aps'][:10]
    model_results = list(apk_norm) + list(apk_nonorm) + \
                    [mrr_norm, mrr_nonorm, mse_norm, mse_nonorm]
    return model_results


def csv_record(mesg, f):
    writer = csv.writer(f)
    writer.writerows(mesg)


def reset_flag(FLAGS, func, str, v):
    delattr(FLAGS, str)
    func(str, v, '')


def reset_graph():
    del_all_flags(tf.flags.FLAGS)
    tf.reset_default_graph()


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


tune(FLAGS, placeholders)
