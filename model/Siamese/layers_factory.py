from config import FLAGS
from layers import GraphConvolution, Average, NTN, Dot, Dense, Padding
import tensorflow as tf
import numpy as np
from math import exp


def create_layers(model):
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
            layers.append(create_Average_layer(layer_info, model))
        elif name == 'NTN':
            layers.append(create_NTN_layer(layer_info, model))
        elif name == 'Dot':
            layers.append(create_Dot_layer(layer_info, model))
        elif name == 'Dense':
            layers.append(create_Dense_layer(layer_info, model))
        elif name == 'Padding':
            layers.append(create_Padding_layer(layer_info, model))
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
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1)


def create_Average_layer(layer_info, model):
    if not len(layer_info) == 0:
        raise RuntimeError('Average layer must have 0 specs')
    return Average()


def create_NTN_layer(layer_info, model):
    if not len(layer_info) == 5:
        raise RuntimeError('Average layer must have 0 specs')
    return NTN(
        input_dim=int(layer_info['input_dim']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']),
        bias=parse_as_bool(layer_info['bias']))


def create_Dot_layer(layer_info, model):
    if not len(layer_info) == 0:
        raise RuntimeError('Dot layer must have 0 specs')
    return Dot()


def create_Dense_layer(layer_info, model):
    if not len(layer_info) == 5:
        raise RuntimeError('Dot layer must have 5 specs')
    return Dense(
        input_dim=int(layer_info['input_dim']),
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']))


def create_Padding_layer(layer_info, model):
    if not len(layer_info) == 2:
        raise RuntimeError('Padding layer must have 2 specs')
    return Padding(
        max_in_dims=int(layer_info['max_in_dims']),
        padding_value=int(layer_info['padding_value']))


def create_activation(act, sim_kernel=None, use_tf=True):
    if act == 'relu':
        return tf.nn.relu if use_tf else relu_np
    elif act == 'identity':
        return tf.identity if use_tf else identity_np
    elif act == 'sigmoid':
        return tf.sigmoid if use_tf else sigmoid_np
    elif act == 'tanh':
        return tf.tanh if use_tf else np.tanh
    elif act == 'sim_kernel':
        return sim_kernel.dist_to_sim_tf if use_tf else \
            sim_kernel.dist_to_sim_np
    else:
        raise RuntimeError('Unknown activation function {}'.format(act))


def relu_np(x):
    return np.maximum(x, 0)


def identity_np(x):
    return x


def sigmoid_np(x):
    return 1 / (1 + exp(-x))


def parse_as_bool(b):
    if b == 'True':
        return True
    elif b == 'False':
        return False
    else:
        raise RuntimeError('Unknown bool string {}'.format(b))
