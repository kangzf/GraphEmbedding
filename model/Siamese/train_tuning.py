from __future__ import division
from __future__ import print_function

from utils_siamese import check_flags
from eval_siamese import Eval
from data_siamese import SiameseModelData
from dist_calculator import DistCalculator
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
FLAGS = flags.FLAGS

def del_all_flags(FLAGS):
 flags_dict = FLAGS._flags()
 keys_list = [keys for keys in flags_dict]
 for keys in keys_list:
  FLAGS.__delattr__(keys)

def reset_graph():
 del_all_flags(tf.flags.FLAGS)
 tf.reset_default_graph()

def train(valid_percentage=0.4, norm_dist=True, yeta=0.2, final_act='identity',learning_rate=0.01, iters=500):
  # For data preprocessing.
  """ dataset: aids50, aids10k, """
  flags.DEFINE_string('dataset', 'aids50', 'Dataset string.')
  """ valid_percentage: (0, 1). """
  flags.DEFINE_float('valid_percentage', valid_percentage,
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
  flags.DEFINE_boolean('norm_dist', norm_dist,
                       'Whether to normalize the distance or not.')
  """ sim_kernel: gaussian. """  # TODO: linear
  flags.DEFINE_string('sim_kernel', 'gaussian',
                      'Name of the similarity kernel.')
  """ yeta: if norm_dist, recommend 0.2; else, try 0.001. """
  flags.DEFINE_float('yeta', yeta, 'yeta for the gaussian kernel function.')
  """ final_act: identity, relu, sigmoid, tanh, sim_kernel (same as sim_kernel). """
  flags.DEFINE_string('final_act', final_act,
                      'The final activation function applied to the NTN output.')
  """ loss_func: mse. """  # TODO: sigmoid pairwise, etc.
  flags.DEFINE_string('loss_func', 'mse', 'Loss function(s) to use.')
  """ sim_kernel: gaussian. """  # TODO: linear
  flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
  flags.DEFINE_float('weight_decay', 5e-4,
                     'Weight for L2 loss on embedding matrix.')

  # For training and validating.
  flags.DEFINE_integer('batch_size', 2, 'Number of graphs in a batch.') # TODO: implement
  flags.DEFINE_float('learning_rate', learning_rate, 'Initial learning rate.')
  flags.DEFINE_integer('iters', iters, 'Number of iterations to train.')
  """ early_stopping: None for no early stopping. """
  flags.DEFINE_integer('early_stopping', None,
                       'Tolerance for early stopping (# of iters).')

  # For testing.
  flags.DEFINE_boolean('plot_results', False,
                       'Whether to plot the results or not.')

  check_flags(FLAGS)

  data = SiameseModelData(FLAGS)

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
      FLAGS, placeholders, input_dim=data.input_dim(), logging=True)

  sess = tf.Session()

  sess.run(tf.global_variables_initializer())


  def run_tf(train_val_test, test_id=None, train_id=None):
      feed_dict = data.get_feed_dict(
          FLAGS, placeholders, dist_calculator, train_val_test, test_id, train_id)
      if train_val_test == 'train':
          objs = [model.opt_op, model.loss]
          # objs = [model.pred_sim(), model.opt_op, model.loss] # TODO: figure out why it's slow
      elif train_val_test == 'val':
          objs = [model.loss]
      elif train_val_test == 'test':
          objs = [model.pred_sim_without_act()]
      else:
          raise RuntimeError('Unknown train_val_test {}'.format(train_val_test))
      t = time()
      outs = sess.run(objs, feed_dict=feed_dict)
      time_rtn = time() - t
      if train_val_test == 'test':
          outs = sess.run([model.pred_sim()], feed_dict=feed_dict)
      return outs[-1], time_rtn


  train_costs, train_times, val_costs, val_times = [], [], [], []
  best_val_loss, best_train_loss = float('Inf'), float('Inf')

  for iter in range(FLAGS.iters):
      # Train.
      train_cost, train_time = run_tf('train')
      train_costs.append(train_cost)
      train_times.append(train_time)

      # Validate.
      val_cost, val_time = run_tf('val')
      val_costs.append(val_cost)
      val_times.append(val_time)

      # Test.
      # test_cost, test_time = run_tf('test')

      print('Iter:', '%04d' % (iter + 1),
            'train_loss=', '{:.5f}'.format(train_cost),
            'time=', '{:.5f}sec'.format(train_time),
            'val_loss=', '{:.5f}'.format(val_cost),
            'time=', '{:.5f}sec'.format(val_time))
      # 'test_loss=', '{:.5f}'.format(test_cost),
      # 'time=', '{:.5f}'.format(test_time))

      if val_cost < best_val_loss:
        best_val_loss = val_cost
        best_val_iter = iter
      if train_cost < best_train_loss:
        best_train_loss = train_cost
        best_train_iter = iter

      if FLAGS.early_stopping:
          if iter > FLAGS.early_stopping and \
                  val_costs[-1] > np.mean(val_costs[-(FLAGS.early_stopping + 1):-1]):
              print('Early stopping...')
              break

  print('Optimization Finished!')

  # Test
  eval = Eval(FLAGS.dataset, FLAGS.sim_kernel, FLAGS.yeta, FLAGS.plot_results)
  m, n = data.m_n()
  test_sim_mat = np.zeros((m, n))
  test_time_mat = np.zeros((m, n))
  run_tf('test', 0, 0) # flush the pipeline
  print('i,j,time,sim,true_sim')
  for i in range(m):
      for j in range(n):
          sim_i_j, test_time = run_tf('test', i, j)
          test_time *= 1000
          print('{},{},{:.2f}mec,{:.2f},{:.2f}'.format(
              i, j, test_time, sim_i_j,
              eval.get_true_sim(i, j, FLAGS.norm_dist)))
          # assert (0 <= sim_i_j <= 1)
          test_sim_mat[i][i] = sim_i_j
          test_time_mat[i][j] = test_time
  print('Evaluating...')
  results = eval.eval_test(FLAGS.model, test_sim_mat, test_time_mat)

  reset_graph()
  # print(results)
  return best_train_loss, best_train_iter, best_val_loss, best_val_iter, results

if __name__ == '__main__':
  train()


