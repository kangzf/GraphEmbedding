'''
Load (and post-process) result data.
'''

from utils import get_root_path
from glob import glob
import numpy as np


class Result(object):
    def ged_mat(self):
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
        self.ged_mat_ = self._get_result_mat(dataset, 'ged')
        self.time_mat_ = self._get_result_mat(dataset, 'time')

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

    def _get_result_mat(self, dataset, metric):
        file_p = get_root_path() + '/files/{}/{}/ged_{}_mat_{}_{}_*.npy'.format( \
            dataset, metric, metric, dataset, self.model_)
        li = glob(file_p)
        if len(li) != 1:
            raise RuntimeError('Files for {}: {}'.format(file_p, li))
        file = li[0]
        return np.load(file)


class EmbeddingBasedModelResult(Result):
    def __init__(self, dataset, model, sim):
        pass


def load_results_as_dict(dataset, models):
    rtn = {}
    for model in models:
        rtn[model] = load_result(dataset, model)
    return rtn

def load_result(dataset, model, sim='dot'):
    if 'beam' in model or model in ['astar', 'hungarian', 'vj']:
        return PairwiseGEDModelResult(dataset, model)
    elif model == 'graph2vec':
        return EmbeddingBasedModelResult(dataset, model, sim)
