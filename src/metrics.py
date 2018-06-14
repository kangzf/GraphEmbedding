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
        assert(len(true_ids) >= 1)
        min_rank = float('inf')
        for true_id in true_ids:
            pred_rank = pred_r.ranking(i, true_id, norm, one_based=True)
            min_rank = min(min_rank, pred_rank)
        topanswer_ranks[i] = min_rank
        if i in print_ids:
            print('query {}\nrank: {}'.format(i, min_rank))
    return 1.0 / hmean(topanswer_ranks)


def ndcg_from_ranking(y_true, ranking):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    ranking : array-like, shape = [k]
        Document indices, i.e.,
            ranking[0] is the index of top-ranked document,
            ranking[1] is the index of second-ranked document,
            ...
    k : int
        Rank.
    Returns
    -------
    NDCG @k : float
    -------
    Credit: https://gist.github.com/mblondel/7337391
    """
    k = len(ranking)
    best_ranking = np.argsort(y_true, kind='mergesort')[::-1]
    best = dcg_from_ranking(y_true, best_ranking[:k])
    return dcg_from_ranking(y_true, ranking) / best


def dcg_from_ranking(y_true, ranking):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    ranking : array-like, shape = [k]
        Document indices, i.e.,
            ranking[0] is the index of top-ranked document,
            ranking[1] is the index of second-ranked document,
            ...
    k : int
        Rank.
    Returns
    -------
    DCG @k : float
    -------
    Credit: https://gist.github.com/mblondel/7337391
    """
    y_true = np.asarray(y_true)
    ranking = np.asarray(ranking)
    rel = y_true[ranking]
    gains = 2 ** rel - 1
    discounts = np.log2(np.arange(len(ranking)) + 2)
    return np.sum(gains / discounts)


if __name__ == '__main__':
    print(ndcg_from_ranking([10, 1, 70], [2, 1, 0]))
    # print(ndcg_from_ranking([2, 2, 3], [2, 1, 0]))
