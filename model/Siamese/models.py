from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS


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
    def __init__(self, placeholders, input_dim, output_dim, yeta, **kwargs):
        super(GCNTN, self).__init__(**kwargs)

        self.inputs_1 = placeholders['features_1']
        self.inputs_2 = placeholders['features_2']
        self.support_1 = placeholders['support_1']
        self.support_2 = placeholders['support_2']
        self.num_features_1_nonzero = placeholders['num_features_1_nonzero']
        self.num_features_2_nonzero = placeholders['num_features_2_nonzero']

        self.num_supports = placeholders['num_supports']

        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = output_dim  # placeholders['labels'].get_shape().as_list()[1]
        self.yeta = yeta
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _build(self):
        with tf.variable_scope(self.name):
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                # num_supports = self.num_supports,
                                                logging=self.logging))

            self.layers.append(Average(placeholders=self.placeholders))

            self.layers.append(NTN(input_dim=FLAGS.hidden1,
                                   feature_map_dim=FLAGS.feature_map_dim,
                                   placeholders=self.placeholders,
                                   act=tf.nn.relu,
                                   dropout=True,
                                   logging=self.logging))

        # Build sequential layer model
        hidden_1 = self.layers[0]([self.inputs_1, self.support_1, self.num_features_1_nonzero])
        hidden_2 = self.layers[0]([self.inputs_2, self.support_2, self.num_features_2_nonzero])
        self.middle_1 = self.layers[1](hidden_1)
        self.middle_2 = self.layers[1](hidden_2)
        self.outputs = self.layers[2]([self.middle_1, self.middle_2])

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def _loss(self):
        # Weight decay loss.
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        # L2 loss.
        self.loss += tf.nn.l2_loss(self.placeholders['labels'] - self.outputs)

    # def name(self):
