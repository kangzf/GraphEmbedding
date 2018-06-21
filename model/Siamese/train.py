from __future__ import division
from __future__ import print_function

from utils_siamese import check_flags, print_msec
from eval_siamese import Eval
from data_siamese import SiameseModelData
from dist_calculator import DistCalculator
from saver import Saver
from models import GCNTN
from time import time
import numpy as np
import tensorflow as tf

# Set random seeds.
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Hyper-parameters.
flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

# For data preprocessing.
""" dataset: aids50, aids10k, """
flags.DEFINE_string('dataset', 'aids50', 'Dataset string.')
""" valid_percentage: (0, 1). """
flags.DEFINE_float('valid_percentage', 0.4,
                   '(# validation graphs) / (# validation + # training graphs.')
""" node_feat_name: 'type' for aids. """
flags.DEFINE_string('node_feat_name', 'type', 'Name of the node feature.')
""" node_feat_encoder: onehot. """  # TODO: 'random', 'random_cond_on_node_feat'
flags.DEFINE_string('node_feat_encoder', 'onehot',
                    'How to encode the node feature.')
""" edge_feat_name: 'valence' for aids. """  # TODO: really use
flags.DEFINE_string('edge_feat_name', 'valence', 'Name of the edge feature.')
""" edge_feat_processor: None. """  # TODO: 'ECC', 'ToNode'
flags.DEFINE_string('edge_feat_processor', None,
                    'How to process the edge feature.')
""" dist_metric: ged. """
flags.DEFINE_string('dist_metric', 'ged', 'Distance metric to use.')
""" dist_algo: beam80 for ged. """
flags.DEFINE_string('dist_algo', 'beam80',
                    'Ground-truth distance algorithm to use.')
""" sampler: random. """  # TODO: density
flags.DEFINE_string('sampler', 'random', 'Sampler to use.')
""" sample_num: 1, 2, 3, ..., -1 (infinite/continuous sampling). """
flags.DEFINE_integer('sample_num', -1,
                     'Number of pairs to sample for training.')
""" sampler_duplicate_removal: False. """  # TODO: True
flags.DEFINE_boolean('sampler_duplicate_removal', False,
                     'Whether to remove duplicate for sampler or not.')

# For model.
""" model: gcntn. """
flags.DEFINE_string('model', 'siamese_gcntn', 'Model string.')
flags.DEFINE_integer('num_layers', 4, 'Number of layers.')
flags.DEFINE_string(
    'layer_0',
    'GraphConvolution:output_dim=32,act=relu,'
    'dropout=True,bias=True,sparse_inputs=True', '')
flags.DEFINE_string(
    'layer_1',
    'GraphConvolution:input_dim=32,output_dim=16,act=identity,'
    'dropout=True,bias=True,sparse_inputs=False', '')
flags.DEFINE_string(
    'layer_2',
    'Average', '')
flags.DEFINE_string(
    'layer_3',
    'NTN:input_dim=16,feature_map_dim=10,inneract=relu,'
    'dropout=True,bias=True', '')
""" norm_dist: True, False. """
flags.DEFINE_boolean('norm_dist', True,
                     'Whether to normalize the distance or not.')
""" sim_kernel: gaussian. """  # TODO: linear
flags.DEFINE_string('sim_kernel', 'gaussian',
                    'Name of the similarity kernel.')
""" yeta: if norm_dist, recommend 0.2; else, try 0.001. """
flags.DEFINE_float('yeta', 0.2, 'yeta for the gaussian kernel function.')
""" final_act: identity, relu, sigmoid, tanh, sim_kernel (same as sim_kernel). """
flags.DEFINE_string('final_act', 'identity',
                    'The final activation function applied to the NTN output.')
""" loss_func: mse. """  # TODO: sigmoid pairwise, etc.
flags.DEFINE_string('loss_func', 'mse', 'Loss function(s) to use.')
""" sim_kernel: gaussian. """  # TODO: linear
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4,
                   'Weight for L2 loss on embedding matrix.')

# For training and validating.
flags.DEFINE_integer('batch_size', 2, 'Number of graphs in a batch.')  # TODO: implement
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('iters', 5, 'Number of iterations to train.')
""" early_stopping: None for no early stopping. """
flags.DEFINE_integer('early_stopping', None,
                     'Tolerance for early stopping (# of iters).')
flags.DEFINE_boolean('log', False,
                     'Whether to log the results via Tensorboard, etc. or not.')

# For testing.
flags.DEFINE_boolean('plot_results', True,
                     'Whether to plot the results or not.')

check_flags(FLAGS)
data = SiameseModelData()
dist_calculator = DistCalculator(FLAGS.dataset, FLAGS.dist_metric,
                                 FLAGS.dist_algo)

if FLAGS.model == 'siamese_gcntn':
    num_supports = 1
    model_func = GCNTN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

placeholders = {
    'support_1': tf.sparse_placeholder(tf.float32),
    'features_1': tf.sparse_placeholder(tf.float32, shape=None),
    'support_2': tf.sparse_placeholder(tf.float32),
    'features_2': tf.sparse_placeholder(tf.float32, shape=None),
    'num_supports': tf.placeholder(tf.int32),
    'dist': tf.placeholder(tf.float32, shape=None),
    'norm_dist': tf.placeholder(tf.float32, shape=None),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_1_nonzero': tf.placeholder(tf.int32),
    'num_features_2_nonzero': tf.placeholder(tf.int32)
}

model = model_func(
    FLAGS, placeholders, input_dim=data.input_dim(), logging=FLAGS.log)
sess = tf.Session()
saver = Saver(FLAGS, sess)
sess.run(tf.global_variables_initializer())


def run_tf(tvt, test_id=None, train_id=None, iter=None):
    feed_dict = data.get_feed_dict(
        placeholders, dist_calculator, tvt, test_id, train_id)
    if tvt == 'train':
        objs = [model.opt_op, model.loss]
        # objs = [model.pred_sim(), model.opt_op, model.loss] # TODO: figure out why it's slow
    elif tvt == 'val':
        objs = [model.loss]
    elif tvt == 'test':
        objs = [model.pred_sim_without_act()]
    else:
        raise RuntimeError('Unknown train_val_test {}'.format(tvt))
    objs = saver.proc_objs(objs, tvt)
    t = time()
    outs = sess.run(objs, feed_dict=feed_dict)
    time_rtn = time() - t
    saver.proc_outs(outs, tvt, iter)
    if tvt == 'test':
        # tf_result = sess.run([model.pred_sim()], feed_dict=feed_dict)[-1]
        # print('tf result', tf_result[-1])
        np_result = model.apply_final_act_np(outs[-1])
        # print('np result', np_result)
        outs[-1] = np_result
    return outs[-1], time_rtn


train_costs, train_times, val_costs, val_times = [], [], [], []
for iter in range(FLAGS.iters):
    # Train.
    train_cost, train_time = run_tf('train', iter=iter)
    train_costs.append(train_cost)
    train_times.append(train_time)

    # Validate.
    val_cost, val_time = run_tf('val', iter=iter)
    val_costs.append(val_cost)
    val_times.append(val_time)

    # Test.
    # test_cost, test_time = run_tf('test')

    print('Iter:', '%04d' % (iter + 1),
          'train_loss=', '{:.5f}'.format(train_cost),
          'time=', print_msec(train_time),
          'val_loss=', '{:.5f}'.format(val_cost),
          'time=', print_msec(val_time))
    # 'test_loss=', '{:.5f}'.format(test_cost),
    # 'time=', '{:.5f}'.format(test_time))

    if FLAGS.early_stopping:
        if iter > FLAGS.early_stopping and \
                val_costs[-1] > np.mean(val_costs[-(FLAGS.early_stopping + 1):-1]):
            print('Early stopping...')
            break

print('Optimization Finished!')

# Test.
eval = Eval(FLAGS.dataset, FLAGS.sim_kernel, FLAGS.yeta, FLAGS.plot_results)
m, n = data.m_n()
test_sim_mat = np.zeros((m, n))
test_time_mat = np.zeros((m, n))
run_tf('test', 0, 0)  # flush the pipeline
print('i,j,time,sim,true_sim')
for i in range(m):
    for j in range(n):
        sim_i_j, test_time = run_tf('test', i, j)
        test_time *= 1000
        print('{},{},{:.2f}mec,{:.4f},{:.4f}'.format(
            i, j, test_time, sim_i_j,
            eval.get_true_sim(i, j, FLAGS.norm_dist)))
        # assert (0 <= sim_i_j <= 1)
        test_sim_mat[i][i] = sim_i_j
        test_time_mat[i][j] = test_time
print('Evaluating...')
results = eval.eval_test(FLAGS.model, test_sim_mat, test_time_mat)
print('Results generated with {} metrics'.format(len(results)))
