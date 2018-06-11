'''
Load (and post-process) result data.
'''

from utils import get_root_path, get_file_base_id, load_data, load_pkl, save_pkl
from glob import glob
import numpy as np
import json, pickle
from os.path import isfile


class Result(object):
    def model(self):
        return self.model_

    def m_n(self):
        raise NotImplementedError()

    def ged_mat(self, norm):
        raise NotImplementedError()

    def sim_mat(self):
        raise NotImplementedError()

    def ged_sim_mat(self, norm):
        raise NotImplementedError()

    def ged_sim(self, qid, gid, norm):
        raise NotImplementedError()

    def top_k_ids(self, qid, k, norm, inclusive):
        ged_sort_id_mat = self.ged_sort_id_mat(norm)
        _, n = ged_sort_id_mat.shape
        if k < 0 or k >= n:
            raise RuntimeError('Invalid k {}'.format(k))
        if not inclusive:
            return ged_sort_id_mat[qid][:k]
        # Tie inclusive.
        ged_sim_mat = self.ged_sim_mat(norm)
        while k < n:
            cid = ged_sort_id_mat[qid][k - 1]
            nid = ged_sort_id_mat[qid][k]
            if ged_sim_mat[qid][cid] == ged_sim_mat[qid][nid]:
                k += 1
            else:
                break
        return ged_sort_id_mat[qid][:k]

    def ranking(self, qid, gid, norm):
        raise NotImplementedError()

    def time_mat(self):
        raise NotImplementedError()

    def time(self, qid, gid):
        raise NotImplementedError()

    def mat(self, metric, norm):
        raise NotImplementedError()

    def ged_sort_id_mat(self, norm):
        raise NotImplementedError()


class PairwiseGEDModelResult(Result):
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model_ = model
        self.ged_mat_ = self._load_result_mat(dataset, 'ged')
        self.ged_norm_mat_ = np.copy(self.ged_mat_)
        train_data = load_data(self.dataset, True)
        test_data = load_data(self.dataset, False)
        m, n = self.ged_mat_.shape
        for i in range(m):
            lm = test_data.graphs[i].number_of_nodes()
            for j in range(n):
                ln = train_data.graphs[j].number_of_nodes()
                self.ged_norm_mat_[i][j] = 2 * self.ged_mat_[i][j] / (lm + ln)
        self.time_mat_ = self._load_result_mat(dataset, 'time')
        self.ged_sort_id_mat_ = np.argsort(self.ged_mat_, kind='mergesort')
        self.ged_norm_sort_id_mat_ = np.argsort(self.ged_norm_mat_, kind='mergesort')

    def m_n(self):
        return self.ged_mat_.shape

    def ged_mat(self, norm):
        return self._select_ged_mat(norm)

    def ged_sim_mat(self, norm):
        return self.ged_mat(norm)

    def ged_sim(self, qid, gid, norm):
        return 'ged', self._select_ged_mat(norm)[qid][gid]

    def ranking(self, qid, gid, norm):
        # Assume self is ground truth.
        ged_sort_id_mat = self.ged_sort_id_mat(norm)
        ged_mat = self.ged_mat(norm)
        finds = np.where(ged_sort_id_mat[qid] == gid)
        assert (len(finds) == 1 and len(finds[0]) == 1)
        fid = finds[0][0]
        # Tie inclusive (always when find ranking).
        while fid > 0:
            cid = ged_sort_id_mat[qid][fid]
            pid = ged_sort_id_mat[qid][fid - 1]
            if ged_mat[qid][pid] == ged_mat[qid][cid]:
                fid -= 1
            else:
                break
        return fid + 1 # 1-based

    def time_mat(self):
        return self.time_mat_

    def time(self, qid, gid):
        return self.time_mat_[qid][gid]

    def mat(self, metric, norm):
        if metric == 'ged':
            return self._select_ged_mat(norm)
        elif metric == 'time':
            return self.time_mat_
        else:
            raise RuntimeError('Unknown metric {} for model {}'.format( \
                metric, self.model_))

    def ged_sort_id_mat(self, norm):
        return self._select_ged_sort_id_mat(norm)

    def _load_result_mat(self, dataset, metric):
        file_p = get_root_path() + '/files/{}/{}/ged_{}_mat_{}_{}_*.npy'.format( \
            dataset, metric, metric, dataset, self.model_)
        li = glob(file_p)
        if len(li) != 1:
            raise RuntimeError('Files for {}: {}'.format(file_p, li))
        file = li[0]
        return np.load(file)

    def _select_ged_mat(self, norm):
        return self.ged_norm_mat_ if norm else self.ged_mat_

    def _select_ged_sort_id_mat(self, norm):
        return self.ged_norm_sort_id_mat_ if norm else self.ged_sort_id_mat_


class EmbeddingBasedModelResult(Result):
    def m_n(self):
        return self.sim_mat_.shape

    def sim_mat(self):
        return self.sim_mat_

    def ged_sim_mat(self, *unused):
        raise self.sim_mat()

    def ged_sim(self, qid, gid, *unused):
        return 'sim', self.sim_mat_[qid][gid]

    def ged_sort_id_mat(self, *unused):
        return np.argsort(self.sim_mat_, kind='mergesort')[:, ::-1]


class Graph2VecResult(EmbeddingBasedModelResult):
    def __init__(self, dataset, model, sim):
        self.dim = 1024
        self.model_ = model
        self.dataset = dataset
        self.sim = sim
        self.sim_mat_ = self._load_sim_mat()
        self.ged_sort_id_mat_ = self.ged_sort_id_mat()

    def mat(self, metric, *unused):
        if metric == 'sim':
            return self.sim_mat_
        else:
            raise RuntimeError('Unknown metric {} for model {}'.format( \
                metric, self.model_))

    def time(self, qid, gid):
        return None

    def _load_sim_mat(self):
        fn = get_root_path() + '/files/{}/sim/{}_graph2vec_dim_{}_sim_{}.npy'.format( \
            self.dataset, self.dataset, self.dim, self.sim)
        if isfile(fn):
            with open(fn, 'rb') as handle:
                sim_mat = load_pkl(handle)
                print('Loaded sim mat from {}'.format(fn))
                return sim_mat
        train_emb = self._load_emb(True)
        test_emb = self._load_emb(False)
        if self.sim == 'dot':
            sim_mat = test_emb.dot(train_emb.T)
        else:
            raise RuntimeError('Unknown sim {}'.format(self.sim))
        with open(fn, 'wb') as handle:
            save_pkl(sim_mat, handle)
            print('Saved sim mat {} to {}'.format(sim_mat.shape, fn))
        return sim_mat

    def _load_emb(self, train):
        fn = get_root_path() + '/files/{}/emb/{}_graph2vec_{}_emb_dim_{}.npy'.format( \
            self.dataset, self.dataset, 'train' if train else 'test', self.dim)
        if isfile(fn):
            emb = np.load(fn)
            print('Loaded emb {} from {}'.format(emb.shape, fn))
            return emb
        data = load_data(self.dataset, train=train)
        id_map = self._gid_to_matrixid(data)
        emb = np.zeros((len(data.graphs), self.dim))
        cnt = 0
        d = self._load_json_emb()
        for f in d:
            gid = get_file_base_id(f)
            if gid in id_map:
                emb[id_map[gid]] = d[f]
                cnt += 1
        if cnt != len(id_map):
            raise RuntimeError('Mismatch: {} != {}').format( \
                cnt, len(id_map))
        np.save(fn, emb)
        print('Saved emb {} to {}'.format(emb.shape, fn))
        return emb

    def _load_json_emb(self):
        fn = get_root_path() + '/save/{}_graph2vec_json_dict.pkl'.format( \
            self.dataset)
        if isfile(fn):
            with open(fn, 'rb') as handle:
                d = load_pkl(handle)
                print('Loaded json dict from {}'.format(fn))
                return d
        dfn = get_root_path() + '/graph2vec_tf/embeddings/{}_train_test_dims_{}_epochs_1000_lr_0.3_embeddings.txt'.format(
            self.dataset, self.dim)
        with open(dfn) as json_data:
            d = json.load(json_data)
        with open(fn, 'wb') as handle:
            save_pkl(d, handle)
            print('Loaded json dict from {}\nSaved to {}'.format(dfn, fn))
        return d

    def _gid_to_matrixid(self, data):
        rtn = {}
        for i, g in enumerate(data.graphs):
            rtn[g.graph['gid']] = i
        return rtn


def load_results_as_dict(dataset, models):
    rtn = {}
    for model in models:
        rtn[model] = load_result(dataset, model)
    return rtn


def load_result(dataset, model, sim='dot'):
    if 'beam' in model or model in ['astar', 'hungarian', 'vj']:
        return PairwiseGEDModelResult(dataset, model)
    elif model == 'graph2vec':
        return Graph2VecResult(dataset, model, sim)


if __name__ == '__main__':
    Graph2VecResult('aids10k', 'graph2vec', sim='dot')
