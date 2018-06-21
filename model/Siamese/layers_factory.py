from layers import GraphConvolution, Average, NTN
import sys
from os.path import dirname, abspath

sys.path.insert(0, "{}/../src".format(dirname(dirname(abspath(__file__)))))
from similarity import create_sim_kernel
import tensorflow as tf


def create_layers(model, FLAGS):
    layers = []
    num_layers = FLAGS.num_layers
    for i in range(num_layers):
        sp = FLAGS.flag_values_dict()['layer_{}'.format(i)].split(':')
        name = sp[0]
        layer_info = {}
        if len(sp) > 1:
            assert (len(sp) == 2)
            for spec in sp[1].split(','):
                ssp = spec.split('=')
                layer_info[ssp[0]] = ssp[1]
        if name == 'GraphConvolution':
            layers.append(create_GraphConvolution_layer(layer_info, model, i))
        elif name == 'Average':
            layers.append(create_Average_layer(layer_info))
        elif name == 'NTN':
            layers.append(create_NTN_layer(layer_info, model))
        else:
            raise RuntimeError('Unknown layer {}'.format(name))
    return layers


def create_GraphConvolution_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 6:
        raise RuntimeError('GraphConvolution layer must have 3-4 specs')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 0:
            raise RuntimeError(
                'The input dim for layer {} must be specified'.format(i))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return GraphConvolution(
        input_dim=input_dim,
        output_dim=int(layer_info['output_dim']),
        placeholders=model.placeholders,
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1,
        logging=model.logging)


def create_Average_layer(layer_info):
    if not len(layer_info) <= 0:
        raise RuntimeError('Average layer must have 0 specs')
    return Average()


def create_NTN_layer(layer_info, model):
    if not len(layer_info) == 5:
        raise RuntimeError('Average layer must have 0 specs')
    return NTN(
        input_dim=int(layer_info['input_dim']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        placeholders=model.placeholders,
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']),
        bias=parse_as_bool(layer_info['bias']),
        logging=model.logging)


def create_activation(act, sim_kernel=None):
    if act == 'relu':
        return tf.nn.relu
    elif act == 'identity':
        return tf.identity
    elif act == 'sigmoid':
        return tf.sigmoid
    elif act == 'tanh':
        return tf.tanh
    elif act == 'sim_kernel':
        return sim_kernel
    else:
        raise RuntimeError('Unknown activation function {}'.format(act))


def parse_as_bool(b):
    if b == 'True':
        return True
    elif b == 'False':
        return False
    else:
        raise RuntimeError('Unknown bool string {}'.format(b))
