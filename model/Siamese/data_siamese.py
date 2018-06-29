from data import Data
from utils import load_data, exec_turnoff_print
from utils_siamese import get_phldr
from samplers import RandomSampler, DistributionSampler
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
import numpy as np
import networkx as nx

exec_turnoff_print()


class SiameseModelData(Data):
    def __init__(self, FLAGS):
        self.dataset = FLAGS.dataset
        self.valid_percentage = FLAGS.valid_percentage
        self.node_feat_name = FLAGS.node_feat_name
        self.node_feat_encoder = FLAGS.node_feat_encoder
        self.edge_feat_name = FLAGS.edge_feat_name
        self.edge_feat_processor = FLAGS.edge_feat_processor
        self.sampler = FLAGS.sampler
        self.sample_num = FLAGS.sample_num
        self.sampler_duplicate_removal = FLAGS.sampler_duplicate_removal
        super().__init__(self._get_name())
        print('{} train graphs; {} validation graphs; {} test graphs'.format( \
            self.train_data.num_graphs(),
            self.valid_data.num_graphs(),
            self.test_data.num_graphs()))

    def init(self):
        orig_train_data = load_data(self.dataset, train=True)
        self.n = len(orig_train_data.graphs)
        train_gs, valid_gs = self._train_val_split(orig_train_data)
        test_gs = load_data(self.dataset, train=False).graphs
        self.node_feat_encoder = self._get_node_feature_encoder( \
            orig_train_data.graphs + test_gs)
        self._check_graphs_num(test_gs, 'test')
        self.train_data = ModelGraphList(
            self.sampler, self.sample_num, self.sampler_duplicate_removal,
            train_gs, self.node_feat_encoder)
        self.valid_data = ModelGraphList(
            self.sampler, self.sample_num, self.sampler_duplicate_removal,
            valid_gs, self.node_feat_encoder)
        self.test_data = ModelGraphList(
            self.sampler, self.sample_num, self.sampler_duplicate_removal,
            test_gs, self.node_feat_encoder)
        self.m = self.test_data.num_graphs()

        assert (len(train_gs) + len(valid_gs) == len(orig_train_data.graphs))

    def input_dim(self):
        return self.node_feat_encoder.input_dim()

    def get_feed_dict(self, FLAGS, phldr, dist_calculator, tvt, \
                      test_id, train_id):
        rtn = dict()
        # no pair is specified == train or val
        if tvt == 'train' or tvt == 'val':
            assert (test_id is None and train_id is None)
            pairs = []
            for _ in range(FLAGS.batch_size):
                pairs.append(self._get_graph_pair(tvt))
        else:
            assert (tvt == 'test')
            g1 = self.test_data.get_graph(test_id)
            g2 = self._get_orig_train_graph(train_id)
            pairs = [(g1, g2)]
        for i, (g1, g2) in enumerate(pairs):
            rtn[get_phldr(phldr, 'inputs_1', tvt)[i]] = \
                g1.get_node_inputs()
            rtn[get_phldr(phldr, 'inputs_2', tvt)[i]] = \
                g2.get_node_inputs()
            rtn[get_phldr(phldr, 'num_inputs_1_nonzero', tvt)[i]] = \
                g1.get_node_inputs_num_nonzero()
            rtn[get_phldr(phldr, 'num_inputs_2_nonzero', tvt)[i]] = \
                g2.get_node_inputs_num_nonzero()
            num_laplacians = 1
            for j in range(num_laplacians):
                rtn[get_phldr(phldr, 'laplacians_1', tvt)[i][j]] = \
                    g1.get_laplacians()[j]
                rtn[get_phldr(phldr, 'laplacians_2', tvt)[i][j]] = \
                    g2.get_laplacians()[j]
                assert (len(g1.get_laplacians()) == len(g2.get_laplacians())
                        == num_laplacians)
            if tvt == 'train' or tvt == 'val':
                dists = np.zeros((FLAGS.batch_size, 1))
                norm_dists = np.zeros((FLAGS.batch_size, 1))
                for i in range(FLAGS.batch_size):
                    g1, g2 = self._get_graph_pair(tvt)
                    dist, norm_dist = self._get_dist(
                        g1.get_nxgraph(), g2.get_nxgraph(), dist_calculator)
                    dists[i] = dist
                    norm_dists[i] = norm_dist
                rtn[phldr['dists']] = dists
                rtn[phldr['norm_dists']] = norm_dists
                rtn[phldr['dropout']] = FLAGS.dropout
        return rtn

    def m_n(self):
        return self.m, self.n

    def _get_name(self):
        li = []
        for k, v in sorted(self.__dict__.items(), key=lambda x: x[0]):
            li.append('{}'.format(v))
        return '_'.join(li)

    def _train_val_split(self, orig_train_data):
        if self.valid_percentage < 0 or self.valid_percentage > 1:
            raise RuntimeError('valid_percentage {} must be in [0, 1]'.format(
                self.valid_percentage))
        gs = orig_train_data.graphs
        sp = int(len(gs) * (1 - self.valid_percentage))
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
        if self.node_feat_encoder == 'onehot':
            return NodeFeatureOneHotEncoder(gs, self.node_feat_name)
        else:
            raise RuntimeError('Unknown node_feat_encoder {}'.format(
                self.node_feat_encoder))

    def _get_graph_collection(self, train_val_test):
        if train_val_test == 'train':
            return self.train_data
        elif train_val_test == 'val':
            return self.valid_data
        elif train_val_test == 'test':
            return self.test_data
        else:
            raise RuntimeError('Unknown train_val_test {}'.format(
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
    def __init__(self, gs, node_feat_name):
        self.node_feat_name = node_feat_name
        inputs_set = set()
        for g in gs:
            inputs_set = inputs_set | set(self._node_feat_dic(g).values())
        self.feat_idx_dic = {feat: idx for idx, feat in enumerate(inputs_set)}
        self.oe = OneHotEncoder().fit(
            np.array(list(self.feat_idx_dic.values())).reshape(-1, 1))

    def encode(self, g):
        node_feat_dic = self._node_feat_dic(g)
        temp = [self.feat_idx_dic[node_feat_dic[n]] for n in g.nodes()]
        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()

    def input_dim(self):
        return self.oe.transform([[0]]).shape[1]

    def _node_feat_dic(self, g):
        return nx.get_node_attributes(g, self.node_feat_name)


class ModelGraphList(object):
    def __init__(self, sampler, sample_num, sampler_duplicate_removal,
                 gs, node_feat_encoder):
        self.gs = [ModelGraph(g, node_feat_encoder) for g in gs]
        if sampler == 'random':
            self.sampler = RandomSampler(
                self.gs, sample_num, sampler_duplicate_removal)
        elif sampler == 'density':
            self.sampler = DistributionSampler(
                self.gs, sample_num, sampler_duplicate_removal)
        else:
            raise RuntimeError('Unknown sampler {}'.format(sampler))

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
        encoded_inputs = node_feat_encoder.encode(nxgraph)
        self.node_inputs = self._preprocess_inputs(
            sp.csr_matrix(encoded_inputs))
        # Only one laplacian, i.e. Laplacian.
        self.laplacians = [self._preprocess_adj(nx.adjacency_matrix(nxgraph))]

    def get_nxgraph(self):
        return self.nxgraph

    def get_node_inputs(self):
        return self.node_inputs

    def get_node_inputs_num_nonzero(self):
        return self.node_inputs[1].shape

    def get_laplacians(self):
        return self.laplacians

    def _preprocess_inputs(self, inputs):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(inputs.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        inputs = r_mat_inv.dot(inputs)
        return self._sparse_to_tuple(inputs)

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
