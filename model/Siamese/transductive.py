import sys
from os.path import dirname, abspath

sys.path.insert(0, '{}/../src'.format(dirname(dirname(abspath(__file__)))))
from utils import load_data, exec_turnoff_print
from similarity import create_sim_kernel
from dist_calculator import DistCalculator
from eval import Eval
import numpy as np
import scipy.sparse as sp

MODEL = 'transductive'
DATASET = 'aids50nef'
DIST_METRIC = 'ged'
DIST_ALGO = 'beam80'
SIM_KERNEL = 'gaussian'
NONORM_YETA = 0.001
NORM_YETA = 0.2
DIMS = [10, 20, 30, 40, 50, 59]


def main():
    exec_turnoff_print()
    sim_mat_dict, ttsp = load_train_test_joint_sim_mat()
    eval_dict = {'nonorm': Eval(DATASET, SIM_KERNEL, NONORM_YETA, plot_results=True),
                 'norm': Eval(DATASET, SIM_KERNEL, NORM_YETA, plot_results=True)}
    for norm_str, sim_mat in sim_mat_dict.items():
        for dim in DIMS:
            emb = perform_svd(sim_mat, dim)
            evaluate_emb(
                emb, ttsp, eval_dict[norm_str],
                '{}_{}_{}'.format(MODEL, norm_str, dim))


def load_train_test_joint_sim_mat():
    train_data = load_data(DATASET, train=True)
    n = len(train_data.graphs)
    test_data = load_data(DATASET, train=False)
    m = len(test_data.graphs)
    t_graphs = train_data.graphs + test_data.graphs  # first train, then test
    tl = m + n
    assert (tl == len(t_graphs))
    # No normalization.
    dist_mat = np.zeros((tl, tl))
    nonorm_sim_mat = np.zeros((tl, tl))
    nonorm_sim_kernel = create_sim_kernel(SIM_KERNEL, NONORM_YETA)
    # Normalization.
    norm_dist_mat = np.zeros((tl, tl))
    norm_sim_mat = np.zeros((tl, tl))
    norm_sim_kernel = create_sim_kernel(SIM_KERNEL, NORM_YETA)
    dist_calculator = DistCalculator(DATASET, DIST_METRIC, DIST_ALGO)
    for i in range(tl):
        for j in range(tl):
            dist_mat[i][j], norm_dist_mat[i][j] = \
                dist_calculator.calculate_dist(t_graphs[i], t_graphs[j])
            nonorm_sim_mat[i][j] = nonorm_sim_kernel.dist_to_sim_np(
                dist_mat[i][j])
            norm_sim_mat[i][j] = norm_sim_kernel.dist_to_sim_np(
                norm_dist_mat[i][j])
    return {'nonorm': nonorm_sim_mat, 'norm': norm_sim_mat}, n


def perform_svd(mat, dim):
    u, s, v = sp.linalg.svds(mat, dim, return_singular_vectors="u")
    # return U \Sigma^{1/2}
    return sp.diags(np.sqrt(s)).dot(u.T).T


def evaluate_emb(emb, ttsp, eval, model):
    train_emb = emb[0:ttsp]
    test_emb = emb[ttsp:]
    pred_sim_mat = test_emb.dot(train_emb.T)
    eval.eval_test(model, pred_sim_mat, None)


if __name__ == '__main__':
    main()
