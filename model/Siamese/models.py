from layers_factory import create_layers, create_activation
from similarity import create_sim_kernel
from utils_siamese import get_phldr
import tensorflow as tf


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'log'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        log = kwargs.get('log', False)
        self.log = log

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
        print('Model built')
        # Build metrics
        self._loss()
        print('Loss built')
        self.opt_op = self.optimizer.minimize(self.loss)
        print('Optimizer built')

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
        self.phldr = placeholders
        self.input_dim = input_dim
        self.batch_size = FLAGS.batch_size
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
        print('Created {} layers: {}'.format(
            len(self.layers), ', '.join(l.get_name() for l in self.layers)))

        # Build the siamese model for train_val and test, respectively,
        # because train_val involve batch_size, but test does not (only 1 pair).
        for tvt in ['train_val', 'test']:
            num_pairs = self.batch_size if tvt == 'train_val' else 1
            log = (tvt == 'train_val')
            # Go through each layer except the last one.
            acts = [[], []]
            for k in range(0, len(self.layers) - 1):
                layer = self.layers[k]
                for j in range(2):
                    ins = self._get_ins(k, layer, num_pairs, j, acts[j], tvt, log)
                    outs = self._get_outs(ins, layer, j, log)
                    assert (type(outs) is list)
                    assert (len(outs) == num_pairs)
                    acts[j].append(outs)
            # Assume the last layer is (and only the last layer is)
            # the merging layer, e.g. NTN, dot.
            merging_layer = self.layers[-1]
            outs = self._call_mergeing_layer(merging_layer, num_pairs, acts, log)
            if tvt == 'train_val':
                self.outputs = outs
            else:
                self.test_output = outs

        # Store model variables for easy access.
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
            l2_loss = tf.nn.l2_loss(
                self.sim_kernel.dist_to_sim_tf(self._get_dist()) -
                self.pred_sim()) / self.batch_size
            self.loss += l2_loss
            tf.summary.scalar('l2_loss', l2_loss)
        else:
            raise RuntimeError('Unknown loss function {}'.format(self.loss_func))

        tf.summary.scalar('total_loss', self.loss)

    # def _rank_loss(self, gamma):
    #     y_pred = self.pred_sim()
    #     pos_interact_score = y_pred[
    #                          :FLAGS.batch_size_p]  # need set new flag for positive sample number & assume pred is a vector
    #     neg_interact_score = y_pred[FLAGS.batch_size_p:]
    #     diff_mat = tf.reshape(tf.tile(pos_interact_score, [FLAGS.num_negatives]),
    #                           # need set new flag for negative sampling
    #                           (-1,
    #                            1)) - neg_interact_score  # assume negative sampling is conducted in this way: p+n1+n2+..+nk
    #     rank_loss = tf.reduce_mean(-tf.log(tf.sigmoid(gamma * diff_mat)))
    #     return rank_loss

    def pred_sim(self):
        return self.final_act(self.outputs)

    def pred_sim_without_act(self):
        return self.test_output

    def apply_final_act_np(self, score):
        return self.final_act_np(score)

    def _get_ins(self, layer_idx, layer, num_pairs, j, prev_outs, tvt, log):
        if layer_idx == 0:
            ins = []
            for i in range(num_pairs):
                ins.append(get_phldr(
                    self.phldr, 'inputs_{}'.format(j + 1), tvt)[i])
        else:
            ins = prev_outs[-1]
            ins_mat = tf.concat(ins, 0)
            self._log_mat(ins_mat, layer, 'ins_{}'.format(j + 1), log)
        if layer.__class__.__name__ == 'GraphConvolution':
            for i in range(len(ins)):
                ins[i] = \
                    [ins[i],
                     self._get_laplacians(i, j, tvt),
                     self._get_num_inputs_nonzero(i, j, tvt)]
        return ins

    def _get_outs(self, ins, layer, j, log):
        outs = []
        for input in ins:
            output = layer(input)
            outs.append(output)
        outs_mat = tf.concat(outs, 0)
        self._log_mat(outs_mat, layer, 'outs_{}'.format(j + 1), log)
        return outs

    def _call_mergeing_layer(self, merging_layer, num_pairs, acts, log):
        ins = []
        outs = []
        for i in range(num_pairs):
            input_0 = acts[0][-1][i]
            input_1 = acts[1][-1][i]
            ins.append(input_0)
            ins.append(input_1)
            output = merging_layer([input_0, input_1])
            outs.append(output)
        ins_mat = tf.stack(ins)
        outs = tf.stack(outs)
        self._log_mat(ins_mat, merging_layer, 'ins', log)
        self._log_mat(outs, merging_layer, 'outs', log)
        return outs

    def _log_mat(self, mat, layer, label, log):
        if log:
            tf.summary.histogram(layer.name + '/' + label, mat)
        pass

    def _get_laplacians(self, i, j, tvt):
        if j == 0:
            return get_phldr(self.phldr, 'laplacians_1', tvt)[i]
        elif j == 1:
            return get_phldr(self.phldr, 'laplacians_2', tvt)[i]
        else:
            assert (False)

    def _get_num_inputs_nonzero(self, i, j, tvt):
        if j == 0:
            return get_phldr(self.phldr, 'num_inputs_1_nonzero', tvt)[i]
        elif j == 1:
            return get_phldr(self.phldr, 'num_inputs_2_nonzero', tvt)[i]
        else:
            assert (False)

    def _get_dist(self):
        if self.FLAGS.dist_norm:
            return self.phldr['norm_dists']
        else:
            return self.phldr['dists']
