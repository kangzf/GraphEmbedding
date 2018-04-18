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
        self.name = name
        sfn = self.save_filename()
        try:
            self.load()
            print('%s loaded from %s' % (name, sfn))
        except:
            self.init()
            self.save()
            print('%s saved to %s' % (name, sfn))
            self.num_graphs = len(self.graphs)
            self.dist = self.get_dist_mat()

    def get_dist_mat(self):
        dist_mat = np.zeros((self.num_graphs, self.num_graphs))
        print('Generating distance matrix of {}'.format(dist_mat.shape))
        print('i,j,#node_i,#node_j,time')
        for i in range(self.num_graphs):
            for j in range(i + 1, self.num_graphs):
                t = time()
                gi = self.graphs[i]
                gj = self.graphs[j]
                ged = GED(gi, gj)
                dist_mat[i][j] = ged
                dist_mat[j][i] = ged
                print('{},{},{},{},{:.5f}'.format( \
                    i, j, len(gi), len(gj), time() - t))
        return dist_mat


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


class SynData(Data):
    def __init__(self, train):
        super().__init__(train)

    def init(self):
        self.graphs = {}
        if self.train:
            num_graphs = 90
        else:
            num_graphs = 10
        for i in range(num_graphs):
            n = randint(5, 20)
            m = randint(n - 1, n * (n - 1) / 2)
            self.graphs[i] = nx.gnm_random_graph(n, m)
        print('Randomly generated %s graphs' % num_graphs)



def play_with_nx():
    g = nx.gnm_random_graph(5, 7)
    print(g)


if __name__ == '__main__':
    play_with_nx()
