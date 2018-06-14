from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn_utils import *
from models import GCNTN
from scipy import sparse

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'aids10k', 'Dataset string.')
flags.DEFINE_integer('sample_num', 2, 'number of samples from train set')
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

adj_train, feature_train, adj_test, feature_test, y_train, y_test, adj_all, feature_all, idx = data_load(FLAGS.dataset, FLAGS.sample_num)

# toy case test
# features_1 = sparse.csr_matrix(np.array([[1.,2.,3.],[3.,4.,5.]]))
# features_2 = sparse.csr_matrix(np.array([[3.,4.,5.],[4.,5.,6.],[7.,8.,9.]]))
# adj_1 = sparse.csr_matrix(np.array([[1.,1.],[1.,1.]]))
# adj_2 = sparse.csr_matrix(np.array([[1.,1.,1.],[1.,1.,0.],[1.,0.,1.]]))
# y_train = np.array([[0.9]])

# Some preprocessing
prefeatures_all = []
prefeatures_test = []
presupport_all = []
presupport_test = []

for f in feature_all:
    prefeatures_all.append(preprocess_features(f))
features_1 = features_2 = [prefeatures_all[i] for i in idx]
for f in feature_test:
    prefeatures_test.append(preprocess_features(f))

if FLAGS.model == 'gcntn':
    for a in adj_all:
        presupport_all.append([preprocess_adj(a)])
    support_1 = support_2 = [presupport_all[i] for i in idx]
    for a in adj_test:
        presupport_test.append([preprocess_adj(a)])

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
    'features_1': tf.sparse_placeholder(tf.float32, shape = None), # shape=tf.constant(features_1[2], dtype=tf.int64)
    'support_2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features_2': tf.sparse_placeholder(tf.float32, shape = None), # shape=tf.constant(features_2[2], dtype=tf.int64)
    'num_supports': tf.placeholder(tf.int32),
    'labels': tf.placeholder(tf.float32, shape = None), # shape=(None, y_train.shape[1])
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
    loss_val = []
    for row, f_s1 in enumerate(zip(features_1, support_1)):
        for col, f_s2 in enumerate(zip(features_2, support_2)):
            feed_dict_val = construct_feed_dict(features_1, features_2, support_1, support_2, labels[row][col], placeholders)
            temp_loss = sess.run(model.loss, feed_dict=feed_dict_val)
            loss_val.append(temp_loss)
    return sum(loss_val)/float(len(loss_val)), (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    cost_train = []
    for row, f_s1 in enumerate(zip(features_1, support_1)):
        for col, f_s2 in enumerate(zip(features_2, support_2)):
            # Construct feed dictionary
            feed_dict = construct_feed_dict(f_s1[0],f_s2[0], f_s1[1], f_s2[1], y_train[row][col], placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)
            cost_train.append(outs[1])

    # Validation
    cost, duration = evaluate(prefeatures_val, prefeatures_val, presupport_val, presupport_val, y_val, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "val_loss=", "{:.5f}".format(cost), "time=", "{:.5f}".format(time.time() - t))

    # Early stopping
    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break

print("Optimization Finished!")

# Testing
test_cost, test_duration = evaluate(prefeatures_test, prefeatures_all, presupport_test, presupport_all, y_test, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost), "time=", "{:.5f}".format(test_duration))





