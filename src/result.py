'''
Load (and post-process) result data.
'''

from utils import get_root_path, get_file_base_id, load_data, load_pkl, save_pkl
from glob import glob
import numpy as np
import json, pickle
from os.path import isfile


class Result(object):
    def ged_mat(self):
        raise NotImplementedError()

    def sim_mat(self):
        raise NotImplementedError()

    def time_mat(self):
        raise NotImplementedError()

    def mat(self, metric):
        raise NotImplementedError()

    def ged_sort_id_mat(self):
        raise NotImplementedError()


class PairwiseGEDModelResult(Result):
    def __init__(self, dataset, model):
        self.model_ = model
        self.ged_mat_ = self._load_result_mat(dataset, 'ged')
        self.time_mat_ = self._load_result_mat(dataset, 'time')

    def ged_mat(self):
        return self.ged_mat_

    def time_mat(self):
        return self.time_mat_

    def mat(self, metric):
        if metric == 'ged':
            return self.ged_mat_
        elif metric == 'time':
            return self.time_mat_
        else:
            raise RuntimeError('Unknown metric {} for model {}'.format( \
                metric, self.model_))

    def ged_sort_id_mat(self):
        return np.argsort(self.ged_mat_)

    def _load_result_mat(self, dataset, metric):
        file_p = get_root_path() + '/files/{}/{}/ged_{}_mat_{}_{}_*.npy'.format( \
            dataset, metric, metric, dataset, self.model_)
        li = glob(file_p)
        if len(li) != 1:
            raise RuntimeError('Files for {}: {}'.format(file_p, li))
        file = li[0]
        return np.load(file)


class EmbeddingBasedModelResult(Result):
    def sim_mat(self):
        return self.sim_mat_

    def ged_sort_id_mat(self):
        return np.argsort(self.sim_mat_)[:,::-1]


class Graph2VecResult(EmbeddingBasedModelResult):
    def __init__(self, dataset, model, sim):
        self.dim = 1024
        self.model = model
        self.dataset = dataset
        self.sim = sim
        self.sim_mat_ = self._load_sim_mat()

    def mat(self, metric):
        if metric == 'sim':
            return self.sim_mat_
        else:
            raise RuntimeError('Unknown metric {} for model {}'.format( \
                metric, self.model_))

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
