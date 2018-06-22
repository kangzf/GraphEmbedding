import tensorflow as tf

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
""" sim_kernel: gaussian, identity. """  # TODO: linear
flags.DEFINE_string('sim_kernel', 'identity',
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
flags.DEFINE_integer('iters', 500, 'Number of iterations to train.')
""" early_stopping: None for no early stopping. """
flags.DEFINE_integer('early_stopping', None,
                     'Tolerance for early stopping (# of iters).')
flags.DEFINE_boolean('log', True,
                     'Whether to log the results via Tensorboard, etc. or not.')

# For testing.
flags.DEFINE_boolean('plot_results', True,
                     'Whether to plot the results or not.')

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
