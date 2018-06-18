import sys
from os.path import dirname, abspath

sys.path.insert(0, "{}/../src".format(dirname(dirname(abspath(__file__)))))
from utils import get_save_path, save, load, exec_turnoff_print
from distance import ged
from collections import OrderedDict


class DistCalculator(object):
    def __init__(self, dataset, dist_metric, algo):
        self.sfn = '{}/{}_{}_{}_gidpair_dist_map'.format( \
            get_save_path(), dataset, dist_metric, algo)
        self.algo = algo
        self.gidpair_dist_map = load(self.sfn)
        if not self.gidpair_dist_map:
            self.gidpair_dist_map = OrderedDict()
            save(self.sfn, self.gidpair_dist_map)
            print('Saved dist map to {} with {} entries'.format( \
                self.sfn, len(self.gidpair_dist_map)))
        else:
            print('Loaded dist map from {} with {} entries'.format( \
                self.sfn, len(self.gidpair_dist_map)))
        if dist_metric == 'ged':
            self.dist_func = ged
        else:
            raise RuntimeError('Unknwon distance metric {}'.format(dist_metric))

    def calculate_dist(self, g1, g2):
        gid1 = g1.graph['gid']
        gid2 = g2.graph['gid']
        pair = (gid1, gid2)
        d = self.gidpair_dist_map.get(pair)
        if d:
            return d
        else:
            rev_pair = (gid2, gid1)
            rev_d = self.gidpair_dist_map.get(rev_pair)
            if rev_d:
                d = rev_d
            else:
                d = self.dist_func(g1, g2, self.algo)
            self.gidpair_dist_map[pair] = d
            print('{}Adding entry ({}, {}) to dist map'.format( \
                ' ' * 80, pair, d))
            save(self.sfn, self.gidpair_dist_map)
            return d
