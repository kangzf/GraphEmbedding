from layers_factory import create_layers, create_activation
from similarity import create_sim_kernel
import tensorflow as tf


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []

        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        self._build()
        # Build metrics
        self._loss()
        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class GCNTN(Model):
    def __init__(self, FLAGS, placeholders, input_dim, **kwargs):
        super(GCNTN, self).__init__(**kwargs)
        self.FLAGS = FLAGS
        self.placeholders = placeholders
        self.inputs_1 = placeholders['features_1']
        self.inputs_2 = placeholders['features_2']
        self.support_1 = placeholders['support_1']
        self.support_2 = placeholders['support_2']
        self.num_supports = placeholders['num_supports']
        self.num_features_1_nonzero = placeholders['num_features_1_nonzero']
        self.num_features_2_nonzero = placeholders['num_features_2_nonzero']
        self.input_dim = input_dim
        self.sim_kernel = create_sim_kernel(
            FLAGS.sim_kernel, FLAGS.yeta)
        self.final_act = create_activation(
            FLAGS.final_act, self.sim_kernel, use_tf=True)
        self.final_act_np = create_activation(
            FLAGS.final_act, self.sim_kernel, use_tf=False)
        self.loss_func = FLAGS.loss_func
        self.weight_decay = FLAGS.weight_decay
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)
        print('input_dim', input_dim)
        self.build()

    def _build(self):
        with tf.variable_scope(self.name):
            self.layers = create_layers(self, self.FLAGS)
        print('Created {} layers'.format(len(self.layers)))

        # Build the siamese model.
        self.activations_list = [[], []]
        for i, inputs in enumerate([self.inputs_1, self.inputs_2]):
            activations = self.activations_list[i]
            activations.append(inputs)
            assert (len(self.layers) >= 2)
            for j in range(0, len(self.layers) - 1):
                layer = self.layers[j]
                inputs_to_layer = activations[-1]
                if layer.__class__.__name__ == 'GraphConvolution':
                    inputs_to_layer = \
                        [inputs_to_layer,
                         self._get_support(i),
                         self._get_num_features_nonzero(i)]
                print('Graph {} through layer {}:{}'.format(
                    i + 1, j + 1, layer.get_name()))
                hidden = layer(inputs_to_layer)
                activations.append(hidden)
        merging_layer = self.layers[-1]
        self.outputs = merging_layer(
            [self.activations_list[0][-1], self.activations_list[1][-1]])
        print('Merging graph 1 and 2 through {}'.format(
            merging_layer.get_name()))

        # Store model variables for easy access
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def _loss(self):
        # Weight decay loss.
        for layer in self.layers:  # TODO: why only layers[0] in gcn?
            for var in layer.vars.values():
                wdl = self.weight_decay * tf.nn.l2_loss(var)
                self.loss += wdl
        tf.summary.scalar('weight_decay_loss', wdl)

        if self.loss_func == 'mse':
            # L2 loss.
            self.temp = self.pred_sim  # TODO: investigate
            l2_loss = tf.nn.l2_loss( \
                self.sim_kernel.dist_to_sim_tf(self._get_dist()) - \
                self.pred_sim())
            self.loss += l2_loss
            tf.summary.scalar('l2_loss', l2_loss)
        else:
            raise RuntimeError('Unknown loss function {}'.format(self.loss_func))
        tf.summary.scalar('total_loss', self.loss)

    def _rank_loss(self, gamma):
        y_pred = self.pred_sim()
        pos_interact_score = y_pred[
                             :FLAGS.batch_size_p]  # need set new flag for positive sample number & assume pred is a vector
        neg_interact_score = y_pred[FLAGS.batch_size_p:]
        diff_mat = tf.reshape(tf.tile(pos_interact_score, [FLAGS.num_negatives]),
                              # need set new flag for negative sampling
                              (-1,
                               1)) - neg_interact_score  # assume negative sampling is conducted in this way: p+n1+n2+..+nk
        rank_loss = tf.reduce_mean(-tf.log(tf.sigmoid(gamma * diff_mat)))
        return rank_loss

    def pred_sim(self):
        return self.final_act(self.outputs)

    def pred_sim_without_act(self):
        return self.outputs

    def apply_final_act_np(self, score):
        return self.final_act_np(score)

    def _get_support(self, i):
        if i == 0:
            return self.support_1
        elif i == 1:
            return self.support_2
        else:
            assert (False)

    def _get_num_features_nonzero(self, i):
        if i == 0:
            return self.num_features_1_nonzero
        elif i == 1:
            return self.num_features_2_nonzero
        else:
            assert (False)

    def _get_dist(self):
        if self.FLAGS.dist_norm:
            return self.placeholders['norm_dist']
        else:
            return self.placeholders['dist']
