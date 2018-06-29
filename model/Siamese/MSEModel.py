from config import FLAGS
from models import Model
from layers_factory import create_activation
from similarity import create_sim_kernel
import numpy as np
import tensorflow as tf


class SiameseGCNTNMSE(Model):
    def __init__(self, input_dim):
        assert(FLAGS.loss_func == 'mse')
        self.input_dim = input_dim
        # Train and validate.
        self.laplacians_1 = \
            [[tf.sparse_placeholder(tf.float32)]
             for _ in range(FLAGS.batch_size)]
        self.laplacians_2 = \
            [[tf.sparse_placeholder(tf.float32)]
             for _ in range(FLAGS.batch_size)]
        self.features_1 = \
            [tf.sparse_placeholder(tf.float32) for _ in range(FLAGS.batch_size)]
        self.features_2 = \
            [tf.sparse_placeholder(tf.float32) for _ in range(FLAGS.batch_size)]
        self.num_nonzero_1 = \
            [tf.placeholder(tf.int32) for _ in range(FLAGS.batch_size)]
        self.num_nonzero_2 = \
            [tf.placeholder(tf.int32) for _ in range(FLAGS.batch_size)]
        self.dists = tf.placeholder(tf.float32, shape=(None, 1))
        self.norm_dists = tf.placeholder(tf.float32, shape=(None, 1))
        self.dropout = tf.placeholder_with_default(0., shape=())
        # Test.
        self.test_laplacians_1 = [[tf.sparse_placeholder(tf.float32)]]
        self.test_laplacians_2 = [[tf.sparse_placeholder(tf.float32)]]
        self.test_features_1 = [tf.sparse_placeholder(tf.float32)]
        self.test_features_2 = [tf.sparse_placeholder(tf.float32)]
        self.test_num_nonzero_1 = [tf.placeholder(tf.int32)]
        self.test_num_nonzero_2 = [tf.placeholder(tf.int32)]
        # Label.
        self.sim_kernel = create_sim_kernel(FLAGS.sim_kernel, FLAGS.yeta)
        # Output.
        self.outs = None
        self.final_act = create_activation(
            FLAGS.final_act, self.sim_kernel, use_tf=True)
        self.final_act_np = create_activation(
            FLAGS.final_act, self.sim_kernel, use_tf=False)
        # Build the model.
        super(SiameseGCNTNMSE, self).__init__()

    def apply_final_act_np(self, score):
        return self.final_act_np(score)

    def get_feed_dict(self, data, dist_calculator, tvt, test_id, train_id):
        rtn = dict()
        # no pair is specified == train or val
        if tvt == 'train' or tvt == 'val':
            assert (test_id is None and train_id is None)
            pairs = []
            for _ in range(FLAGS.batch_size):
                pairs.append(data.get_graph_pair(tvt))
        else:
            assert (tvt == 'test')
            g1 = data.test_data.get_graph(test_id)
            g2 = data.get_orig_train_graph(train_id)
            pairs = [(g1, g2)]
        for i, (g1, g2) in enumerate(pairs):
            rtn[self._get_plhdr('features_1', tvt)[i]] = \
                g1.get_node_inputs()
            rtn[self._get_plhdr('features_2', tvt)[i]] = \
                g2.get_node_inputs()
            rtn[self._get_plhdr('num_nonzero_1', tvt)[i]] = \
                g1.get_node_inputs_num_nonzero()
            rtn[self._get_plhdr('num_nonzero_2', tvt)[i]] = \
                g2.get_node_inputs_num_nonzero()
            num_laplacians = 1
            for j in range(num_laplacians):
                rtn[self._get_plhdr('laplacians_1', tvt)[i][j]] = \
                    g1.get_laplacians()[j]
                rtn[self._get_plhdr('laplacians_2', tvt)[i][j]] = \
                    g2.get_laplacians()[j]
                assert (len(g1.get_laplacians()) == len(g2.get_laplacians())
                        == num_laplacians)
            if tvt == 'train' or tvt == 'val':
                dists = np.zeros((FLAGS.batch_size, 1))
                norm_dists = np.zeros((FLAGS.batch_size, 1))
                for i in range(FLAGS.batch_size):
                    g1, g2 = data.get_graph_pair(tvt)
                    dist, norm_dist = data.get_dist(
                        g1.get_nxgraph(), g2.get_nxgraph(), dist_calculator)
                    dists[i] = dist
                    norm_dists[i] = norm_dist
                rtn[self.dists] = dists
                rtn[self.norm_dists] = norm_dists
                rtn[self.dropout] = FLAGS.dropout
        return rtn

    def _get_ins(self, layer, tvt):
        assert (layer.__class__.__name__ == 'GraphConvolution')
        ins = []
        for features in (self._get_plhdr('features_1', tvt) +
                         self._get_plhdr('features_2', tvt)):
            ins.append(features)
        return ins

    def _proc_ins(self, ins, k, layer, tvt):
        ln = layer.__class__.__name__
        if k != 0 and tvt == 'train_val':
            # sparse matrices (k == 0; the first layer) cannot be logged.
            need_log = True
        else:
            need_log = False
        if ln == 'GraphConvolution':
            ins = self._supply_laplacians_etc(ins, tvt)
            if need_log:
                ins_mat = self._stack_concat([i[0] for i in ins])
        else:
            ins_mat = self._stack_concat(ins)
            if ln == 'Dot' or ln == 'NTN':
                assert (len(ins) % 2 == 0)
                proc_ins = []
                i = 0
                j = len(ins) // 2
                for _ in range(len(ins) // 2):
                    proc_ins.append([ins[i], ins[j]])
                    i += 1
                    j += 1
                ins = proc_ins
        if need_log:
            self._log_mat(ins_mat, layer, 'ins')
        return ins

    def _proc_outs(self, outs, k, layer, tvt):
        if tvt == 'train_val':
            outs_mat = self._stack_concat(outs)
            self._log_mat(outs_mat, layer, 'outs')
        return outs

    def _stack_concat(self, list_of_tensors):
        assert(list_of_tensors)
        s = list_of_tensors[0].get_shape()
        if s != ():
            return tf.concat(list_of_tensors, 0)
        else:
            return tf.stack(list_of_tensors)

    def _mse_loss(self):
        dists = self.norm_dists if FLAGS.dist_norm else self.dists
        assert (len(self.train_val_outputs) == FLAGS.batch_size)
        return tf.nn.l2_loss(
            self.sim_kernel.dist_to_sim_tf(dists) -
            self.final_act(self.train_val_outputs)) / FLAGS.batch_size

    def _supply_laplacians_etc(self, ins, tvt):
        for i, (laplacian, num_nonzero) in \
                enumerate(zip(
                    self._get_plhdr('laplacians_1', tvt) +
                    self._get_plhdr('laplacians_2', tvt),
                    self._get_plhdr('num_nonzero_1', tvt) +
                    self._get_plhdr('num_nonzero_2', tvt))):
            ins[i] = [ins[i], laplacian, num_nonzero]
        return ins

    def _get_plhdr(self, key, tvt):
        if tvt == 'train' or tvt == 'val' or tvt == 'train_val':
            return self.__dict__[key]
        else:
            assert (tvt == 'test')
            return self.__dict__['test_' + key]
