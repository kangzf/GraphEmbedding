# AUTO TUNING PARAMETERS
# Hyperparameters   | Range        | Note
# ------------------------------------------------------------------------------
# dist_norm          T/F            2 norm/no norm
# yeta               0.2/0.01       4/6 diff yetas
# activation func    5              5 diff activation layers
# learning_rate      0.06-0.001     6 diff lr
# iteration          500-2000       4 diff iter
# validation ratio   0.2-0.4        2 diff ratio
# dropout            0.0-1.0        2 diff values
# ------------------------------------------------------------------------------
from utils_siamese import get_siamese_dir, get_model_info_as_str
from utils import get_ts
from main import main
from config import FLAGS
import numpy as np
import tensorflow as tf
import csv
import itertools

dataset = 'aids80nef'
file_name = '{}/logs/parameter_tuning_{}_{}.csv'.format(
    get_siamese_dir(), dataset, get_ts())

header = ['dist_norm', 'yeta', 'final_act', 'learning_rate', 'iter',
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

dist_norm_range = [True, False]
yeta_range_norm = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
yeta_range_nonorm = [0.01, 0.005, 0.001, 0.0005]
final_act_range = ['identity', 'relu', 'sigmoid', 'tanh', 'sim_kernel']
lr_range = [0.06, 0.03, 0.01, 0.006, 0.003, 0.001]
iter_range = list(range(500, 2001, 500))
val_ratio_range = [0.25, 0.4]
dropout_range = [0.1, 0.5]


def tune(FLAGS):
    print('Remember to clean up "../../save/SiameseModelData*" '
          'if something does not work!')
    f = setup_file()
    best_results_train_loss, best_results_val_loss = float('Inf'), float('Inf')
    results_train = []
    results_val = []
    i = 1
    for dist_norm in dist_norm_range:
        yeta_range = yeta_range_norm if dist_norm == True else yeta_range_nonorm
        for yeta, final_act, lr, iteration, val_ratio, dropout in \
                itertools.product(
                    yeta_range, final_act_range, lr_range, iter_range,
                    val_ratio_range, dropout_range):
            print('Number of tuning iteration: {}'.format(i),
                  'dist_norm: {}'.format(dist_norm), 'yeta: {}'.format(yeta),
                  'final_act: {}'.format(final_act),
                  'learning_rate: {}'.format(lr),
                  'iteration: {}'.format(iteration),
                  'validation_ratio: {}'.format(val_ratio),
                  'dropout: {}'.format(dropout))
            i += 1
            flags = tf.app.flags
            reset_flag(FLAGS, flags.DEFINE_string, 'dataset', dataset)
            reset_flag(FLAGS, flags.DEFINE_float, 'valid_percentage',
                       val_ratio)
            reset_flag(FLAGS, flags.DEFINE_string, 'model', 'siamese_gcntn_mse')
            reset_flag(FLAGS, flags.DEFINE_integer, 'num_layers', 4)
            reset_flag(
                FLAGS, flags.DEFINE_string,
                'layer_0',
                'GraphConvolution:output_dim=32,act=relu,'
                'dropout=True,bias=True,sparse_inputs=True')
            reset_flag(
                FLAGS, flags.DEFINE_string,
                'layer_1',
                'GraphConvolution:input_dim=32,output_dim=16,act=identity,'
                'dropout=True,bias=True,sparse_inputs=False')
            reset_flag(
                FLAGS, flags.DEFINE_string,
                'layer_2',
                'Average')
            reset_flag(
                FLAGS, flags.DEFINE_string,
                'layer_3',
                'NTN:input_dim=16,feature_map_dim=10,inneract=relu,'
                'dropout=True,bias=True')
            reset_flag(FLAGS, flags.DEFINE_bool, 'dist_norm', dist_norm)
            reset_flag(FLAGS, flags.DEFINE_float, 'yeta', yeta)
            reset_flag(FLAGS, flags.DEFINE_string, 'final_act', final_act)
            reset_flag(FLAGS, flags.DEFINE_float, 'learning_rate', lr)
            reset_flag(FLAGS, flags.DEFINE_integer, 'iters', iteration)
            reset_flag(FLAGS, flags.DEFINE_float, 'dropout', dropout)
            reset_flag(FLAGS, flags.DEFINE_bool, 'log', False)
            reset_flag(FLAGS, flags.DEFINE_bool, 'plot_results', False)
            FLAGS = tf.app.flags.FLAGS
            train_costs, train_times, val_costs, val_times, results \
                = main()
            best_train_loss = np.min(train_costs)
            best_train_iter = np.argmin(train_costs)
            best_val_loss = np.min(val_costs)
            best_val_iter = np.argmin(val_costs)
            print('best_train_loss: {}'.format(best_train_loss),
                  'best_val_loss: {}'.format(best_val_loss),
                  'best_train_iter: {}'.format(best_train_iter),
                  'best_val_iter: {}'.format(best_val_iter))

            model_results = parse_results(results)
            csv_record([str(x) for x in
                        [dist_norm, yeta, final_act, lr, iteration,
                         val_ratio,
                         best_train_loss, best_train_iter, best_val_loss,
                         best_val_iter] + model_results], f)

            if best_train_loss < best_results_train_loss:
                best_results_train_loss = best_train_loss
                results_train = [dist_norm, yeta, final_act, lr,
                                 iteration, val_ratio,
                                 best_train_loss, best_train_iter] + \
                                model_results

            if best_val_loss < best_results_val_loss:
                best_results_val_loss = best_val_loss
                results_val = [dist_norm, yeta, final_act, lr,
                               iteration, val_ratio,
                               best_val_loss, best_val_iter] + model_results

    print(results_train)
    print(results_val)

    csv_record(['Final results:'], f)
    csv_record(['dist_norm', 'yeta', 'final_act', 'learning_rate', 'iter',
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
                'mrr_nonorm', 'mse_norm', 'mse_nonorm'], f)
    csv_record([str(x) for x in results_train], f)

    csv_record(['dist_norm', 'yeta', 'final_act', 'learning_rate', 'iter',
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
                'mrr_nonorm', 'mse_norm', 'mse_nonorm'], f)
    csv_record([str(x) for x in results_val], f)


def setup_file():
    f = open(file_name, 'w')
    f.write(get_model_info_as_str())
    f.write(','.join(map(str, header)) + '\n')
    f.flush()
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
    f.write(','.join(map(str, mesg)) + '\n')
    f.flush()


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


tune(FLAGS)
