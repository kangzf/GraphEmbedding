from utils import get_root_path
from distance import GED
import pickle
import networkx as nx
from random import randint
import numpy as np
from time import time


class Data(object):
    def __init__(self, train):
        name = self.__class__.__name__ + '_'
        self.train = True if train else False
        if train:
            name += 'train'
        else:
            name += 'test'
        name += ('_' + self.name_suffix())
        self.name = name
        sfn = self.save_filename()
        try:
            self.load()
            print('%s loaded from %s' % (name, sfn))
        except Exception as e:
            print(e)
            self.init()
            self.num_graphs = len(self.graphs)
            if not self.train:
                self.test_train_dist = self.get_truth_dist()
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

    def get_truth_dist(self):
        train_data = self.__class__(train=True)
        return self.get_dist_mat(self.graphs, train_data.graphs)

    def get_dist_mat(self, graphs1, graphs2):
        dist_mat = np.zeros((len(graphs1), len(graphs2)))
        print('Generating distance matrix of {}'.format(dist_mat.shape))
        print('i,j,#node_i,#node_j,dist,time')
        for i in range(len(graphs1)):
            for j in range(len(graphs2)):
                t = time()
                gi = graphs1[i]
                gj = graphs2[j]
                ged = GED(gi, gj)
                dist_mat[i][j] = ged
                print('{},{},{},{},{},{:.5f}'.format( \
                    i, j, len(gi), len(gj), ged, time() - t))
        return dist_mat


class SynData(Data):
    ########## parameters
    train_num_graphs = 10
    test_num_graphs = 5
    ####################

    def __init__(self, train):
        if train:
            self.num_graphs = SynData.train_num_graphs
        else:
            self.num_graphs = SynData.test_num_graphs
        super().__init__(train)

    def init(self):
        self.graphs = {}
        for i in range(self.num_graphs):
            n = randint(5, 20)
            m = randint(n - 1, n * (n - 1) / 2)
            self.graphs[i] = nx.gnm_random_graph(n, m)
        print('Randomly generated %s graphs' % self.num_graphs)
        if self.train:
            self.train_train_dist = self.get_dist_mat(self.graphs, self.graphs)

    def name_suffix(self):
        return '{}_{}'.format(SynData.train_num_graphs, SynData.test_num_graphs)


def play_with_nx():
    g = nx.gnm_random_graph(5, 7)
    print(g)


if __name__ == '__main__':
    play_with_nx()
