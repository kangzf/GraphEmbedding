import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys

import random
from os.path import dirname, abspath, exists
sys.path.insert(0, "{}/../src".format(dirname(dirname(abspath(__file__)))))
from utils import load_data
from distance import ged


random.seed(123)

# TODO_1: Sampling based on graph density distribution
def sampling(graphs, sample_num):
    # measure graph density
    # density = [nx.density[g] for g in graphs]
    # idx = [] 

    # TODO: Get the sample index based on distribution
    idx = random.sample(range(0, len(graphs)), sample_num) 
    return [graphs[i] for i in idx], idx

def all_node_feature(graph_list):
    f_set = set()
    for g in graph_list:
        f_set = f_set | set(nx.get_node_attributes(g,'type').values())
    return f_set

def one_hot_encode(f_set):
    from sklearn.preprocessing import OneHotEncoder
    dic = {k: v for v, k in enumerate(f_set)}
    oe = OneHotEncoder().fit(np.array(list(dic.values())).reshape(-1,1))
    return dic, oe

def check_dir(path):
    if not exists(path):
        from os import mkdir
        mkdir(path)

def extract_features(dic, oe, graph_list):
    adj = []
    feature = []
    for g in graph_list:
        adj.append(nx.adjacency_matrix(g))
        temp_attr = nx.get_node_attributes(g,'type')
        temp_mat =  oe.transform(np.array([dic[temp_attr[i]] for i in g.nodes()]).reshape(-1,1)).toarray()
        feature.append(sp.csr_matrix(temp_mat))
    return adj, feature

def save_obj(obj, file_path):
    with open(file_path, 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load_obj(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)

# GED_cal GED_sym_cal can be same func
def GED_cal(graphs1, graphs2, pre_mat=None): # prd
    ged_mat = np.zeros((len(graphs_1),len(graphs2)))
    for row, g1 in enumerate(graphs1):
        for col, g2 in enumerate(graphs2):
            ged_temp = ged(g1, g2, 'beam80')
            if ged_temp[0] == -1: continue
            ged_mat[row][col] = ged_temp[0]
    return ged_mat

def GED_sym_cal(graphs, pre_mat=None): # prd
    ged_mat = np.zeros((len(graphs),len(graphs)))
    for row, g1 in enumerate(graphs):
        for col, g2 in enumerate(graphs):
            if col<=row: continue
            ged_temp = ged(g1, g2, 'beam80')
            if ged_temp[0] == -1: continue
            ged_mat[row][col] = ged_mat[col][row] = ged_temp[0]
    return ged_mat

def data_load(dataset_str, sample_num):
    train = load_data(dataset_str, train=True)
    test = load_data(dataset_str, train=False)

    train_sam, idx = sampling(train.graphs, sample_num)

    check_dir(dirname(abspath(__file__))+'/'+dataset_str)
    # Parse node feature pool, save & One hot encoding node feature
    save_path = dirname(abspath(__file__))+'/'+dataset_str+'/encode'
    check_dir(save_path)
    if not exists(save_path+'/node_feature.pkl'):
        f_set = all_node_feature(train.graphs + test.graphs)
        save_obj(f_set, save_path+'/node_feature.pkl')
    else:
        f_set = load_obj(save_path+'/node_feature.pkl')

    dic, oe = one_hot_encode(f_set)
    input_dim = oe.transform([[0]]).shape[1]

    # Extract graph features and save
    # lists of scipy sparse matrices
    save_path = dirname(abspath(__file__))+'/'+dataset_str+'/extract_features'
    check_dir(save_path)
    if not exists(save_path+'/train_adj.pkl'):
        adj_all, feature_all = extract_features(dic, oe, train.graphs)
        save_obj(adj_all, save_path+'/train_adj.pkl')
        save_obj(feature_all, save_path+'/train_feature.pkl')
    else:
        adj_all = load_obj(save_path+'/train_adj.pkl')
        feature_all = load_obj(save_path+'/train_feature.pkl')

    adj_train = [adj_all[i] for i in idx]
    feature_train = [feature_all[i] for i in idx]

    if not exists(save_path+'/test_adj.pkl'):
        adj_test, feature_test = extract_features(dic, oe, test.graphs)
        save_obj(adj_test, save_path+'/test_adj.pkl')
        save_obj(feature_test, save_path+'/test_feature.pkl')
    else:
        adj_test = load_obj(save_path+'/test_adj.pkl')
        feature_test = load_obj(save_path+'/test_feature.pkl')
    
    # GED ground truth calculation and save
    save_path = dirname(abspath(__file__))+'/'+dataset_str+'/GED'
    check_dir(save_path)
    if not exists(save_path+'/test_GED.pkl'): # Add test GED file name calculated by Yunsheng Bai
        y_test = GED_cal(test.graphs, train.graphs)
        save_obj(y_test, save_path+'/test_GED.pkl')
    else:
        y_test = load_obj(save_path+'/test_GED.pkl')

    # TODO_2: incremental for efficiency based on max_sample
    # max_sample number, so no need to cal the first max_sample GED
    max_sample = 0 if not exists(save_path+'/max_sample.pkl') else load_obj(save_path+'/max_sample.pkl')

    if not exists(save_path+"/train_GED"+str(sample_num)+".pkl"):
        y_train = GED_sym_cal(train_sam)
        save_obj(y_train,save_path+'/train_GED'+str(sample_num)+'.pkl')
    else:
        y_train = load_obj(save_path+'/train_GED'+str(sample_num)+'.pkl')

    return adj_train, feature_train, adj_test, feature_test, y_train, y_test, adj_all, feature_all, idx, input_dim

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data_origin(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
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


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features_1,features_2, support_1, support_2, labels, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features_1']: features_1})
    feed_dict.update({placeholders['features_2']: features_2})

    feed_dict.update({placeholders['support_1'][i]: support_1[i] for i in range(len(support_1))})
    feed_dict.update({placeholders['support_2'][i]: support_2[i] for i in range(len(support_2))})
    feed_dict.update({placeholders['num_supports']: len(support_1)})
    feed_dict.update({placeholders['num_features_1_nonzero']: features_1[1].shape})
    feed_dict.update({placeholders['num_features_2_nonzero']: features_2[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
