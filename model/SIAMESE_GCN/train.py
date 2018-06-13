from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCNTN
from scipy import sparse

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcntn', 'Model string.')  # 'gcntn', gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000000000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('feature_map_dim', 10, 'Number of feature maps in NTN.')
flags.DEFINE_integer('batch_size', 2, 'Number of graph pairs in a batch.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_boolean('mini_batch', False, 'Use Mini_batch')

# Load data
# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

# TODO: load networkx graph dataset
# adj_1, adj_2, features_1, features_2, y_train, y_val, y_test = data_load(FLAGS.dataset)

features_1 = np.array([[1.,2.,3.],[3.,4.,5.]])
features_2 = np.array([[3.,4.,5.],[4.,5.,6.],[7.,8.,9.]])
adj_1 = np.array([[1.,1.],[1.,1.]])
adj_2 = np.array([[1.,1.,1.],[1.,1.,0.],[1.,0.,1.]])
y_train = np.array([[0.9]])

features_1 = sparse.csr_matrix(features_1)
features_2 = sparse.csr_matrix(features_2)
adj_1 = sparse.csr_matrix(adj_1)
adj_2 = sparse.csr_matrix(adj_2)


# Some preprocessing
features_1 = preprocess_features(features_1)
features_2 = preprocess_features(features_2)

if FLAGS.model == 'gcntn':
    support_1 = [preprocess_adj(adj_1)]
    support_2 = [preprocess_adj(adj_2)]
    num_supports = 1
    model_func = GCNTN
# elif FLAGS.model == 'gcn_cheby':
#     support = chebyshev_polynomials(adj, FLAGS.max_degree)
#     num_supports = 1 + FLAGS.max_degree
#     model_func = GCN
# elif FLAGS.model == 'dense':
#     support = [preprocess_adj(adj)]  # Not used
#     num_supports = 1
#     model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support_1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features_1': tf.sparse_placeholder(tf.float32, shape=tf.constant(features_1[2], dtype=tf.int64)),
    'support_2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features_2': tf.sparse_placeholder(tf.float32, shape=tf.constant(features_2[2], dtype=tf.int64)),
    'num_supports': tf.placeholder(tf.int32),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features_1[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features_1, features_2, support_1, support_2, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features_1, features_2, support_1, support_2, labels, placeholders)
    outs_val = sess.run(model.loss, feed_dict=feed_dict_val)
    return outs_val, (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features_1,features_2, support_1, support_2, y_train, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

    # Validation
    cost, duration = evaluate(features_1, features_2, support_1, support_2, y_train, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "val_loss=", "{:.5f}".format(cost), "time=", "{:.5f}".format(time.time() - t))

    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break

print("Optimization Finished!")

# Testing
# test_cost, test_duration = evaluate(features_1, features_2, support_1, support_2, y_test, placeholders)
# print("Test set results:", "cost=", "{:.5f}".format(test_cost), "time=", "{:.5f}".format(test_duration))





