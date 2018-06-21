import numpy as np
from scipy.stats import hmean


class Metric(object):
    def __init__(self, name, ylabel):
        self.name = name
        self.ylabel = ylabel

    def __str__(self):
        return self.name


def precision_at_ks(true_r, pred_r, norm, ks, print_ids=[]):
    """
    Ranking-based. Prec@ks.
    :param true_r: result object indicating the ground truth.
    :param pred_r: result object indicating the prediction.
    :param norm: whether to normalize the results or not.
    :param ks:
    :param print_ids: (optional) The ids of the query to print for debugging.
    :return: list of floats indicating the average precision at different ks.
    """
    m, n = true_r.m_n()
    assert (true_r.m_n() == pred_r.m_n())
    ps = np.zeros((m, len(ks)))
    for i in range(m):
        for k_idx, k in enumerate(ks):
            assert (type(k) is int and k > 0 and k < n)
            true_ids = true_r.top_k_ids(i, k, norm, inclusive=True)
            pred_ids = pred_r.top_k_ids(i, k, norm, inclusive=False)
            ps[i][k_idx] = len(set(true_ids).intersection(set(pred_ids))) / k
        if i in print_ids:
            print('query {}\nks:    {}\nprecs: {}'.format(i, ks, ps[i]))
    return np.mean(ps, axis=0)


def mean_reciprocal_rank(true_r, pred_r, norm, print_ids=[]):
    """
    Ranking based. MRR.
    :param true_r: result object indicating the ground truth.
    :param pred_r: result object indicating the prediction.
    :param norm: whether to normalize the results or not.
    :param print_ids: (optional) the ids of the query to print for debugging.
    :return: float indicating the mean reciprocal rank.
    """
    m, n = true_r.m_n()
    assert (true_r.m_n() == pred_r.m_n())
    topanswer_ranks = np.zeros(m)
    for i in range(m):
        # There may be multiple graphs with the same dist/sim scores
        # as the top answer by the true_r model.
        # Select one with the lowest (minimum) rank
        # predicted by the pred_r model for mrr calculation.
        true_ids = true_r.top_k_ids(i, 1, norm, inclusive=True)
        assert (len(true_ids) >= 1)
        min_rank = float('inf')
        for true_id in true_ids:
            pred_rank = pred_r.ranking(i, true_id, norm, one_based=True)
            min_rank = min(min_rank, pred_rank)
        topanswer_ranks[i] = min_rank
        if i in print_ids:
            print('query {}\nrank: {}'.format(i, min_rank))
    return 1.0 / hmean(topanswer_ranks)


def mean_squared_error(true_r, pred_r, sim_kernel, yeta, norm):
    """
    Regression-based. L2 difference between the ground-truth similarities
        and the predicted similarities.
    :param true_r:
    :param pred_r:
    :param sim_kernel:
    :param yeta:
    :param norm:
    :return:
    """
    m, n = true_r.m_n()
    assert (true_r.m_n() == pred_r.m_n())
    return np.linalg.norm( \
        true_r.sim_mat(sim_kernel, yeta, norm) - \
        pred_r.sim_mat(sim_kernel, yeta, norm)) / (m * n)


def average_time(r):
    """
    :param true_r: 
    :param pred_r: 
    :return: 
    """
    return np.mean(r.time_mat())


if __name__ == '__main__':
    x = np.array([[14, 40, 33, 14, 28]])
    y = np.array([[11, 37, 36, 9, 30]])
    from distance import gaussian_kernel

    sim_x = gaussian_kernel(x, 1.0)
    sim_y = gaussian_kernel(y, 1.0)
    print('sim_x', sim_x)
    print('sim_y', sim_y)
    r = np.linalg.norm(sim_x - sim_y)
    print(r)
    r = np.linalg.norm(np.array([[0, 1], [2, -1]]) - np.array([[0, 1], [0, 1]]))
    print(r)
