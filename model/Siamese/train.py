from __future__ import division
from __future__ import print_function

from siamese_utils import check_flags
from siamese_data import ModelData
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
""" dataset: aids50, aids10k, """
flags.DEFINE_string('dataset', 'aids50', 'Dataset string.')
""" node_feat_name: 'type' for aids. """
flags.DEFINE_string('node_feat_name', 'type', 'Name of the node feature.')
""" node_feat_encoder: onehot. """
flags.DEFINE_string('node_feat_encoder', 'onehot', 'How to encode the node feature.')
""" edge_feat_name: 'valence' for aids. """
flags.DEFINE_string('edge_feat_name', 'valence', 'Name of the edge feature.')
""" edge_feat_processor: None, 'ECC', 'ToNode'. """
flags.DEFINE_string('edge_feat_processor', None, 'How to process the edge feature.')
""" dist_metric: ged. """
flags.DEFINE_string('dist_metric', 'ged', 'Distance metric to use.')
""" sampler: random. """
flags.DEFINE_string('sampler', 'random', 'Sampler to use.')
""" sample_num: 1, 2, 3, ..., None (infinite/continuous sampling). """
flags.DEFINE_integer('sample_num', None, 'Number of pairs to sample for training.')
""" valid_percentage: (0, 1). """
flags.DEFINE_float('valid_percentage', 0.1, \
                   '(# validation graphs) / (# validation + # training graphs.')
""" model: gcntn. """
flags.DEFINE_string('model', 'gcntn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000000000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('feature_map_dim', 10, 'Number of feature maps in NTN.')
flags.DEFINE_integer('batch_size', 2, 'Number of graph pairs in a batch.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
# early_stopping: None for no early stopping.
flags.DEFINE_integer('early_stopping', None, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('yeta', 1, 'yeta for the gaussian kernel function.')

check_flags(FLAGS)

data = ModelData()

if FLAGS.model == 'gcntn':
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
    'labels': tf.placeholder(tf.float32, shape=None),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_1_nonzero': tf.placeholder(tf.int32),
    'num_features_2_nonzero': tf.placeholder(tf.int32)
}

model = model_func(placeholders, input_dim=data.input_dim(), \
                   output_dim=FLAGS.hidden1, yeta=FLAGS.yeta,
                   logging=True)

sess = tf.Session()

sess.run(tf.global_variables_initializer())


def run_tf(train_val_test):
    feed_dict = data.get_feed_dict(placeholders, train_val_test)
    if train_val_test == 'train':
        objs = [model.opt_op, model.loss]
    elif train_val_test == 'val':
        objs = [model.loss]
    else:
        raise RuntimeError('Unknown train_val_test {}'.format(train_val_test))
    t = time()
    outs = sess.run(objs, feed_dict=feed_dict)
    return outs[-1], time() - t


train_costs, train_times, val_costs, val_times = [], [], [], []
for epoch in range(FLAGS.epochs):
    # Train.
    train_cost, train_time = run_tf('train')
    train_costs.append(train_cost)
    train_times.append(train_time)

    # Validation.
    val_cost, val_time = run_tf('val')
    val_costs.append(val_cost)
    val_times.append(val_time)

    print("Epoch:", '%04d' % (epoch + 1), \
          "train_loss=", "{:.5f}".format(train_cost), \
          "time=", "{:.5f}".format(train_time), \
          "val_loss=", "{:.5f}".format(val_cost), \
          "time=", "{:.5f}".format(val_time))

    if FLAGS.early_stopping:
        if epoch > FLAGS.early_stopping and val_cost[-1] > np.mean(val_cost[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

print("Optimization Finished!")

# TODO: test.
