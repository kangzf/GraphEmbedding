import sys
from os.path import dirname, abspath

sys.path.insert(0, "{}/../src".format(dirname(dirname(abspath(__file__)))))
from data import Data
from utils import load_data, exec_turnoff_print
from samplers import RandomSampler
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
import numpy as np
import networkx as nx
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

exec_turnoff_print()


class SiameseModelData(Data):
    def __init__(self):
        super().__init__('{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format( \
            FLAGS.dataset, FLAGS.valid_percentage, FLAGS.node_feat_name, \
            FLAGS.node_feat_encoder, FLAGS.edge_feat_name, \
            FLAGS.edge_feat_processor, FLAGS.dist_metric, FLAGS.dist_algo, \
            FLAGS.sampler, FLAGS.sample_num, FLAGS.sampler_duplicate_removal))
        print('{} train graphs; {} validation graphs; {} test graphs'.format( \
            self.train_data.num_graphs(), \
            self.valid_data.num_graphs(), \
            self.test_data.num_graphs()))

    def init(self):
        orig_train_data = load_data(FLAGS.dataset, train=True)
        self.n = len(orig_train_data.graphs)
        self.node_feat_encoder = self._get_node_feature_encoder( \
            orig_train_data.graphs)
        train_gs, valid_gs = self._train_val_split(orig_train_data)
        test_gs = load_data(FLAGS.dataset, train=False).graphs
        self._check_graphs_num(test_gs, 'test')
        self.train_data = ModelGraphList(train_gs, self.node_feat_encoder)
        self.valid_data = ModelGraphList(valid_gs, self.node_feat_encoder)
        self.test_data = ModelGraphList(test_gs, self.node_feat_encoder)
        self.m = self.test_data.num_graphs()

        assert (len(train_gs) + len(valid_gs) == len(orig_train_data.graphs))

    def input_dim(self):
        return self.node_feat_encoder.input_dim()

    def get_feed_dict(self, placeholders, dist_calculator, tvt, \
                      test_id, train_id):
        feed_dict = dict()
        # no pair is specified == train or val
        if test_id is None or train_id is None:
            assert (test_id is None and train_id is None)
            g1, g2 = self._get_graph_pair(tvt)
            dist, normalized_dist = self._get_dist(
                g1.get_nxgraph(), g2.get_nxgraph(), dist_calculator)
            feed_dict[placeholders['dist']] = dist
            feed_dict[placeholders['norm_dist']] = normalized_dist
        else:
            g1 = self.test_data.get_graph(test_id)
            g2 = self._get_orig_train_graph(train_id)
            # No need to feed the labels.
        feed_dict[placeholders['features_1']] = g1.get_node_features()
        feed_dict[placeholders['features_2']] = g2.get_node_features()
        num_support = 1
        # for i in range(num_support):
        feed_dict[placeholders['support_1']] = g1.get_supports()  # TODO: turn into batching
        feed_dict[placeholders['support_2']] = g2.get_supports()
        feed_dict[placeholders['num_supports']] = len(g1.get_supports())
        assert (len(g1.get_supports()) == len(g2.get_supports()))
        feed_dict[placeholders['num_features_1_nonzero']] = \
            g1.get_node_features()[1].shape  # TODO: refactor
        feed_dict[placeholders['num_features_2_nonzero']] = \
            g2.get_node_features()[1].shape
        if tvt == 'train' or tvt == 'val':
            feed_dict[placeholders['dropout']] = FLAGS.dropout
        return feed_dict

    def m_n(self):
        return self.m, self.n

    def _train_val_split(self, orig_train_data):
        if FLAGS.valid_percentage < 0 or FLAGS.valid_percentage > 1:
            raise RuntimeError('valid_percentage {} must be in [0, 1]'.format( \
                FLAGS.valid_percentage))
        gs = orig_train_data.graphs
        sp = int(len(gs) * (1 - FLAGS.valid_percentage))
        train_graphs = gs[0:sp]
        valid_graphs = gs[sp:]
        self._check_graphs_num(train_graphs, 'train')
        self._check_graphs_num(valid_graphs, 'validation')
        return train_graphs, valid_graphs

    def _check_graphs_num(self, graphs, label):
        if len(graphs) <= 2:
            raise RuntimeError('Insufficient {} graphs {}'.format( \
                label, len(graphs)))

    def _get_node_feature_encoder(self, gs):
        if FLAGS.node_feat_encoder == 'onehot':
            return NodeFeatureOneHotEncoder(gs)
        else:
            raise RuntimeError('Unknown node_feat_encoder {}'.format( \
                FLAGS.node_feat_encoder))

    def _get_graph_collection(self, train_val_test):
        if train_val_test == 'train':
            return self.train_data
        elif train_val_test == 'val':
            return self.valid_data
        elif train_val_test == 'test':
            return self.test_data
        else:
            raise RuntimeError('Unknown train_val_test {}'.format( \
                train_val_test))

    def _get_graph_pair(self, train_val_test):
        graph_collection = self._get_graph_collection(train_val_test)
        return graph_collection.get_graph_pair()

    def _get_dist(self, g1, g2, dist_calculator):
        return dist_calculator.calculate_dist(g1, g2)

    def _get_orig_train_graph(self, orig_train_id):
        trainlen = self.train_data.num_graphs()
        vallen = self.valid_data.num_graphs()
        if 0 <= orig_train_id < trainlen:
            return self.train_data.get_graph(orig_train_id)
        elif orig_train_id < trainlen + vallen:
            return self.valid_data.get_graph(orig_train_id - trainlen)
        else:
            assert (False)


class NodeFeatureOneHotEncoder(object):
    def __init__(self, gs):
        features_set = set()
        for g in gs:
            features_set = features_set | set(self._node_feat_dic(g).values())
        self.feat_idx = {feat: idx for idx, feat in enumerate(features_set)}
        self.oe = OneHotEncoder().fit( \
            np.array(list(self.feat_idx.values())).reshape(-1, 1))

    def encode(self, g):
        node_feat_dic = self._node_feat_dic(g)
        temp = [self.feat_idx[node_feat_dic[n]] for n in g.nodes()]
        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()

    def input_dim(self):
        return self.oe.transform([[0]]).shape[1]

    def _node_feat_dic(self, g):
        return nx.get_node_attributes(g, FLAGS.node_feat_name)


class ModelGraphList(object):
    def __init__(self, gs, node_feat_encoder):
        self.gs = [ModelGraph(g, node_feat_encoder) for g in gs]
        if FLAGS.sampler == 'random':
            self.sampler = RandomSampler( \
                self.gs, FLAGS.sample_num, FLAGS.sampler_duplicate_removal)
        else:
            raise RuntimeError('Unknown sampler {}'.format(FLAGS.sampler))

    def num_graphs(self):
        return len(self.gs)

    def get_graph_pair(self):
        return self.sampler.get_pair()

    def get_graph(self, id):
        return self.gs[id]

    def num_graphs(self):
        return len(self.gs)


class ModelGraph(object):
    def __init__(self, nxgraph, node_feat_encoder):
        self.nxgraph = nxgraph
        encoded_features = node_feat_encoder.encode(nxgraph)
        self.node_features = self._preprocess_features( \
            sp.csr_matrix(encoded_features))
        self.supports = self._preprocess_adj(nx.adjacency_matrix(nxgraph))

    def get_nxgraph(self):
        return self.nxgraph

    def get_node_features(self):
        return self.node_features

    def get_supports(self):
        return self.supports

    def _preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return self._sparse_to_tuple(features)

    def _preprocess_adj(self, adj):
        """Preprocessing of adjacency matrix and conversion to tuple representation."""
        adj_normalized = self._normalize_adj(adj + sp.eye(adj.shape[0]))
        return self._sparse_to_tuple(adj_normalized)

    def _normalize_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def _sparse_to_tuple(self, sparse_mx):
        """Convert sparse matrix to tuple representation."""

        def to_tuple(mx):
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
            return coords, values, shape

        if isinstance(sparse_mx, list):
            for i in range(len(sparse_mx)):
                sparse_mx[i] = to_tuple(sparse_mx[i])
        else:
            sparse_mx = to_tuple(sparse_mx)

        return sparse_mx
