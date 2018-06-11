from utils import get_root_path, sorted_nicely
from distance import ged
import pickle
import networkx as nx
from random import randint
import numpy as np
from time import time
from glob import glob


class Data(object):
    def __init__(self, train):
        name = self.__class__.__name__ + '_'
        self.train = True if train else False
        if train:
            name += 'train'
        else:
            name += 'test'
        name += self.name_suffix()
        self.name = name
        sfn = self.save_filename()
        try:
            self.load()
            print('%s loaded from %s' % (name, sfn))
        except Exception as e:
            self.init()
            self.num_graphs = len(self.graphs)
            self.save()
            print('%s saved to %s' % (name, sfn))

    def name_suffix(self):
        return ''

    def save(self):
        file = open(self.save_filename(), 'wb')
        file.write(pickle.dumps(self.__dict__))
        file.close()

    def load(self):
        file = open(self.save_filename(), 'rb')
        dp = file.read()
        file.close()
        self.__dict__ = pickle.loads(dp)

    def save_filename(self):
        return '{}/save/{}.pkl'.format(get_root_path(), self.name)

    def get_dist_mat(self, graphs1, graphs2):
        dist_mat = np.zeros((len(graphs1), len(graphs2)))
        print('Generating distance matrix of {}'.format(dist_mat.shape))
        print('i,j,#node_i,#node_j,dist,time')
        for i in range(len(graphs1)):
            for j in range(len(graphs2)):
                t = time()
                gi = graphs1[i]
                gj = graphs2[j]
                d = ged(gi, gj, 'beam80')
                dist_mat[i][j] = d
                print('{},{},{},{},{},{:.5f}'.format( \
                    i, j, len(gi), len(gj), d, time() - t))
        return dist_mat

    def get_gids(self):
        return [g.graph['gid'] for g in self.graphs]


class SynData(Data):
    def __init__(self, train):
        if train:
            self.num_graphs = SynData.train_num_graphs
        else:
            self.num_graphs = SynData.test_num_graphs
        super().__init__(train)

    def init(self):
        self.graphs = []
        for i in range(self.num_graphs):
            n = randint(5, 20)
            m = randint(n - 1, n * (n - 1) / 2)
            g = nx.gnm_random_graph(n, m)
            g.graph['gid'] = i
            self.graphs.append(g)
        print('Randomly generated %s graphs' % self.num_graphs)
        if self.train:
            self.train_train_dist = self.get_dist_mat(self.graphs, self.graphs)

    def name_suffix(self):
        return '_{}_{}'.format(SynData.train_num_graphs,
                               SynData.test_num_graphs)


class AIDS10kData(Data):
    def __init__(self, train):
        super().__init__(train)

    def init(self):
        self.graphs = []
        datadir = get_root_path() + '/data/AIDS10k/' + ('train' if self.train \
            else 'test')
        for file in sorted_nicely(glob(datadir + '/*.gexf')):
            gid = int(file.split('/')[-1].split('.')[0])
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            self.graphs.append(g)
            if not nx.is_connected(g):
                raise RuntimeError('{} not connected'.format(gid))
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))





