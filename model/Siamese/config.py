import tensorflow as tf

# Hyper-parameters.
flags = tf.app.flags

# For data preprocessing.
""" dataset: aids80nef, aids700nef, aids10knef. """
flags.DEFINE_string('dataset', 'aids80nef', 'Dataset string.')
""" valid_percentage: (0, 1). """
flags.DEFINE_float('valid_percentage', 0.25,
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
""" dist_algo: beam80, astar for ged. """
flags.DEFINE_string('dist_algo', 'astar',
                    'Ground-truth distance algorithm to use.')
""" sampler: random, density """
flags.DEFINE_string('sampler', 'random', 'Sampler to use.')
""" sample_num: 1, 2, 3, ..., -1 (infinite/continuous sampling). """
flags.DEFINE_integer('sample_num', -1,
                     'Number of pairs to sample for training.')
""" sampler_duplicate_removal: False. """  # TODO: True
flags.DEFINE_boolean('sampler_duplicate_removal', False,
                     'Whether to remove duplicate for sampler or not.')

# For model.
""" model: siamese_gcntn_mse, siamese_gcntn_hinge, siamese_tranductive_ntn. """
flags.DEFINE_string('model', 'siamese_gcntn_mse', 'Model string.')
# flags.DEFINE_integer('num_layers', 1, 'Number of layers.')
# flags.DEFINE_string(
#     'layer_0',
#     'NTN:input_dim=16,feature_map_dim=10,inneract=relu,'
#     'dropout=True,bias=True', '')
flags.DEFINE_integer('num_layers', 5, 'Number of layers.')
flags.DEFINE_string(
    'layer_0',
    'GraphConvolution:output_dim=32,act=relu,'
    'dropout=True,bias=True,sparse_inputs=True', '')
flags.DEFINE_string(
    'layer_1',
    'GraphConvolution:input_dim=32,output_dim=16,act=identity,'
    'dropout=True,bias=True,sparse_inputs=False', '')
# flags.DEFINE_string(
#     'layer_2',
#     'Average', '')
flags.DEFINE_string(
    'layer_2',
    'Dense:input_dim=16,output_dim=1,dropout=True,'
    'act=relu,bias=True', '')
flags.DEFINE_string(
    'layer_3',
    'Padding:max_in_dims=10,padding_value=0', '')  # Assume the max node # is max_in_dim
flags.DEFINE_string(
    'layer_4',
    'NTN:input_dim=10,feature_map_dim=10,inneract=relu,'
    'dropout=True,bias=True', '')
# flags.DEFINE_string(
#     'layer_3',
#     'Dot', '')
flags.DEFINE_integer('batch_size', 5, 'Number of graph pairs in a batch.')  # TODO: implement
""" dist_norm: True, False. """
flags.DEFINE_boolean('dist_norm', True,
                     'Whether to normalize the distance or not '
                     'when choosing the ground truth distance.')
""" sim_kernel: gaussian, identity. """  # TODO: linear
flags.DEFINE_string('sim_kernel', 'gaussian',
                    'Name of the similarity kernel.')
""" yeta: 
 if norm_dist, try 0.6 for nef small, 0.3 for nef, 0.2 for regular;
 else, try 0.01 for nef, 0.001 for regular. """
flags.DEFINE_float('yeta', 0.6, 'yeta for the gaussian kernel function.')
""" final_act: identity, relu, sigmoid, tanh, sim_kernel (same as sim_kernel). """
flags.DEFINE_string('final_act', 'sim_kernel',
                    'The final activation function applied to the NTN output.')

""" loss_func: mse, hinge. """  # TODO: hinge, sigmoid pairwise, etc.
flags.DEFINE_string('loss_func', 'mse', 'Loss function(s) to use.')

# Start of hinge loss.
""" delta and gamma: depend on whether dist_norm is True of False. """
flags.DEFINE_float('delta', 0.1,
                   'Margin between positive pairs ground truth scores'
                   'and negative pairs scores  ground truth scores')
flags.DEFINE_float('gamma', 0.1,
                   'Margin between positive pairs prediction scores'
                   'and negative pairs prediction scores')
flags.DEFINE_integer('num_neg', 5, 'Number of negative samples.')
# End of hinge loss.

""" graph_loss: 1st, None. """ # TODO: 1st.
flags.DEFINE_string('graph_loss', None, 'Loss function(s) to use.')

""" sim_kernel: gaussian. """  # TODO: linear
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4,
                   'Weight for L2 loss on embedding matrix.')
""" learning_rate: 0.01 recommended. """  # TODO: why 0.06 weird?
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

# For training and validating.
flags.DEFINE_integer('iters', 20, 'Number of iterations to train.')
""" early_stopping: None for no early stopping. """
flags.DEFINE_integer('early_stopping', None,
                     'Tolerance for early stopping (# of iters).')
flags.DEFINE_boolean('log', True,
                     'Whether to log the results via Tensorboard or not.')

# For testing.
flags.DEFINE_boolean('plot_results', False,
                     'Whether to plot the results '
                     '(involving all baselines) or not.')

FLAGS = tf.app.flags.FLAGS
