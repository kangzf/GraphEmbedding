from samplers import RandomSampler
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import numpy as np


class ModelGraphList(object):
    def __init__(self, sampler, sample_num, sampler_duplicate_removal,
                 gs, node_feat_encoder):
        self.gs = [ModelGraph(g, node_feat_encoder) for g in gs]
        if sampler == 'random':
            self.sampler = RandomSampler(
                self.gs, sample_num, sampler_duplicate_removal)
        else:
            raise RuntimeError('Unknown sampler {}'.format(sampler))

    def num_graphs(self):
        return len(self.gs)

    def get_graph_pair(self):
        return self.sampler.get_pair()

    def get_graph(self, id):
        return self.gs[id]


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
