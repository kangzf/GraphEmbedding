from config import FLAGS
from layers_factory import create_layers
import tensorflow as tf


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        self.vars = {}
        self.placeholders = {}
        self.layers = []
        self.train_val_outputs = None
        self.test_output = None
        self.loss = 0
        self.optimizer = None
        self.opt_op = None

        self.batch_size = FLAGS.batch_size
        self.loss_func = FLAGS.loss_func
        self.weight_decay = FLAGS.weight_decay
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate)

        self._build()
        print('Model built')
        # Build metrics
        self._loss()
        print('Loss built')
        self.opt_op = self.optimizer.minimize(self.loss)
        print('Optimizer built')

    def _build(self):
        # Create layers according to FLAGS.
        with tf.variable_scope(self.name):
            self.layers = create_layers(self)
        assert (len(self.layers) > 0)
        print('Created {} layers: {}'.format(
            len(self.layers), ', '.join(l.get_name() for l in self.layers)))

        # Build the siamese model for train_val and test, respectively,
        for tvt in ['train_val', 'test']:
            # Go through each layer except the last one.
            acts = [self._get_ins(self.layers[0], tvt)]
            outs = None
            for k, layer in enumerate(self.layers):
                ins = self._proc_ins(acts[-1], k, layer, tvt)
                outs = layer(ins)
                outs = self._proc_outs(outs, k, layer, tvt)
                acts.append(outs)
            if tvt == 'train_val':
                self.train_val_outputs = outs
            else:
                self.test_output = outs

        # Store model variables for easy access.
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def _loss(self):
        # Weight decay loss.
        wdl = 0.0
        for layer in self.layers:  # TODO: why only layers[0] in gcn?
            for var in layer.vars.values():
                wdl = self.weight_decay * tf.nn.l2_loss(var)
                self.loss += wdl
        tf.summary.scalar('weight_decay_loss', wdl)

        if self.loss_func == 'mse':
            # L2 loss.
            l2_loss = self._mse_loss()
            self.loss += l2_loss
            tf.summary.scalar('l2_loss', l2_loss)
        elif self.loss_func == 'hinge':
            hinge_loss = self._hinge_loss()
            self.loss += hinge_loss
            tf.summary.scalar('hinge_loss', hinge_loss)
        else:
            raise RuntimeError('Unknown loss function {}'.format(self.loss_func))

        tf.summary.scalar('total_loss', self.loss)

    def pred_sim_without_act(self):
        return self.test_output

    def apply_final_act_np(self, score):
        raise NotImplementedError()

    def get_feed_dict(self, data, dist_calculator, tvt, test_id, train_id):
        raise NotImplementedError()

    def _get_ins(self, layer, tvt):
        raise NotImplementedError()

    def _proc_ins(self, ins, k, layer, tvt):
        raise NotImplementedError()

    def _proc_outs(self, outs, k, layer, tvt):
        raise NotImplementedError()

    def _mse_loss(self):
        raise NotImplementedError()

    def _hinge_loss(self):
        raise NotImplementedError()

    def _log_mat(self, mat, layer, label):
        if FLAGS.log:
            tf.summary.histogram(layer.name + '/' + label, mat)

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
