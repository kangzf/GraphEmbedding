from inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=True, act=tf.nn.relu, bias=False,
                 featureless=False, num_supports=1, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.support = None

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(num_supports):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs[0]
        self.support = inputs[1]

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        # one pair comparison
        if not FLAGS.mini_batch:
            supports = list()
            for i in range(len(self.support)):
                if not self.featureless:
                    pre_sup = dot(x, self.vars['weights_' + str(i)],
                                  sparse=self.sparse_inputs)
                else:
                    pre_sup = self.vars['weights_' + str(i)]
                support = dot(self.support[i], pre_sup, sparse=True)
                supports.append(support)
            output = tf.add_n(supports)

        # mini-batch
        else:
            supports = list()
            for i in range(len(self.support)):
                if not self.featureless:
                    support = tf.einsum('aij,jk->aik',tf.matmul(support,x),self.vars['weights_' + str(i)])
                else:
                    support = tf.einsum('aij,jk->aik',support,self.vars['weights_' + str(i)]) # if featureless, should init W as N*D
                supports.append(support)
            output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']            

        return self.act(output)


class Average(Layer):
    """Dense layer."""
    def __init__(self, placeholders, **kwargs):
        super(Average, self).__init__(**kwargs)

    def _call(self, inputs):
        x = inputs
        output = tf.reduce_mean(x, 0) # x is N*D

        return output

class NTN(Layer):
    """Dense layer."""
    def __init__(self, input_dim, feature_map_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=True, yeta=1, **kwargs):
        super(NTN, self).__init__(**kwargs)

        self.sparse_inputs = sparse_inputs
        self.feature_map_dim = feature_map_dim
        self.bias = bias
        self.yeta = yeta

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_W'] = glorot([input_dim, input_dim, feature_map_dim],
                                                    name='weights_W')
            self.vars['weights_V'] = glorot([feature_map_dim, input_dim*2],
                                                    name='weights_V')
            self.vars['weights_U'] = glorot([feature_map_dim, 1],
                                                    name='weights_U')
            if self.bias:
                self.vars['bias'] = zeros([feature_map_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        # x_1 = tf.sparse_to_dense(inputs[0]) 
        # x_2 = tf.sparse_to_dense(inputs[1])
        x_1 = inputs[0]
        x_2 = inputs[1]

        # dropout
        if self.sparse_inputs:
            x_1 = sparse_dropout(x_1, 1-self.dropout, self.num_features_nonzero)
            x_2 = sparse_dropout(x_2, 1-self.dropout, self.num_features_nonzero)
        else:
            x_1 = tf.nn.dropout(x_1, 1-self.dropout)
            x_2 = tf.nn.dropout(x_2, 1-self.dropout)

        # one pair comparison
        if not FLAGS.mini_batch:

            x_1 = tf.reshape(x_1, [1,-1])
            x_2 = tf.reshape(x_2, [1,-1])

            feature_map = []
            for i in range(self.feature_map_dim):
                V_out = tf.matmul(tf.reshape(self.vars['weights_V'][i],[1,-1]),tf.concat([tf.transpose(x_1),tf.transpose(x_2)],0))
                temp = tf.matmul(x_1,self.vars['weights_W'][:,:,i])
                h = tf.reduce_sum(temp*x_2) # h = K.sum(temp*e2,axis=1)
                if self.bias:
                    middle = V_out+h+self.vars['bias'][i]
                else:
                    middle = V_out+h
                feature_map.append(middle)

            tensor_bi_product = tf.stack(feature_map) # axis=0
            tensor_bi_product = self.act(tensor_bi_product)

            output = tf.reduce_sum(self.vars['weights_U']*tensor_bi_product)

        # mini-batch
        else:
            temp = tf.einsum('ij,ajk->iak',x_1,self.vars['weights_W'])
            h = tf.squeeze(tf.matmul(temp,tf.expand_dims(x_2, 2)))
            V_out = tf.matmul(tf.concat([x_1,x_2],1),tf.transpose(self.vars['weights_V']))

            if self.bias:
                tensor_bi_product = V_out+h+self.vars['bias']
            else:
                tensor_bi_product = V_out+h

            tensor_bi_product = self.act(tensor_bi_product)

            output = tf.matmul(tensor_bi_product,self.vars['weights_U']) # B*1 labels should be same dim as well
     
        # tensor_bi_product = self.U*self.activation(K.reshape(
        #                     tensor_bi_product,(self.k,batch_size))).T

        return tf.exp(-self.yeta*tf.square(output))






