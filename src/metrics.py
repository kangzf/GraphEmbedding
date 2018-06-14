import numpy as np


class Metric(object):
    def __init__(self, name, ylabel):
        self.name = name
        self.ylabel = ylabel

    def __str__(self):
        return self.name


def precision_at_ks(true_r, pred_r, norm, ks, print_ids=[]):
    '''
    :param true_r: Result object indicating the ground truth.
    :param pred_r: Result object indicating the prediction.
    :param norm: Whether to normalize the results or not.
    :param ks:
    :param print_ids: (optional) The ids of the query to print for debugging.
    :return: A list of floats indicating the average precision at different ks.
    '''
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
