from utils import get_train_str, get_data_path, get_save_path, sorted_nicely, \
    save, load
import networkx as nx
import random
from random import randint
from glob import glob


class Data(object):
    def __init__(self, name_str):
        name = self.__class__.__name__ + '_' + name_str + self.name_suffix()
        self.name = name
        sfn = self.save_filename()
        temp = load(sfn)
        if temp:
            self.__dict__ = temp
            print('%s loaded from %s' % (name, sfn))
        else:
            self.init()
            save(sfn, self.__dict__)
            print('%s saved to %s' % (name, sfn))

    def init(self):
        raise NotImplementedError()

    def name_suffix(self):
        return ''

    def save_filename(self):
        return '{}/{}'.format(get_save_path(), self.name)

    def get_gids(self):
        return [g.graph['gid'] for g in self.graphs]


class SynData(Data):
    train_num_graphs = 20
    test_num_graphs = 10

    def __init__(self, train):
        if train:
            self.num_graphs = SynData.train_num_graphs
        else:
            self.num_graphs = SynData.test_num_graphs
        super().__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        for i in range(self.num_graphs):
            n = randint(5, 20)
            m = randint(n - 1, n * (n - 1) / 2)
            g = nx.gnm_random_graph(n, m)
            g.graph['gid'] = i
            self.graphs.append(g)
        print('Randomly generated %s graphs' % self.num_graphs)

    def name_suffix(self):
        return '_{}_{}'.format(SynData.train_num_graphs,
                               SynData.test_num_graphs)


class AIDSData(Data):
    def __init__(self, train):
        self.train = train
        super().__init__(get_train_str(train))

    def init(self):
        self.graphs = []
        datadir = '{}/{}/{}'.format(
            get_data_path(), self.get_folder_name(), 'train' if self.train \
                else 'test')
        for file in self.sort()(glob(datadir + '/*.gexf')):
            gid = int(file.split('/')[-1].split('.')[0])
            g = nx.read_gexf(file)
            g.graph['gid'] = gid
            self.graphs.append(g)
            if not nx.is_connected(g):
                raise RuntimeError('{} not connected'.format(gid))
        print('Loaded {} graphs from {}'.format(len(self.graphs), datadir))
        if 'nef' in self.get_folder_name():
            print('Removing edge features')
            for g in self.graphs:
                self._remove_valence(g)

    def get_folder_name(self):
        raise NotImplementedError()

    def sort(self):
        raise NotImplementedError()

    def _remove_valence(self, g):
        for n1, n2, d in g.edges(data=True):
            d.pop('valence', None)


class AIDS10kData(AIDSData):
    def get_folder_name(self):
        return 'AIDS10k'

    def sort(self):
        return sorted_nicely


class AIDS10kNEFData(AIDS10kData):
    def init(self):
        self.graphs = AIDS10kData(self.train).graphs
        for g in self.graphs:
            self._remove_valence(g)
        print('Processed {} graphs: valence removed'.format(len(self.graphs)))


class AIDS50NEFData(AIDS10kData):
    def init(self):
        self.graphs = []
        for g in AIDS10kData(self.train).graphs:
            if g.number_of_nodes() <= 12:
                self.graphs.append(g)
        random.Random(123).shuffle(self.graphs)
        if self.train:
            self.graphs = self.graphs[0:50]
        else:
            self.graphs = self.graphs[0:10]
        for g in self.graphs:
            self._remove_valence(g)
        print('Processed {} graphs: valence removed'.format(len(self.graphs)))


class AIDS700nefData(AIDSData):
    def get_folder_name(self):
        return 'AIDS700nef'

    def sort(self):
        return sorted_nicely

    def _remove_valence(self, g):
        for n1, n2, d in g.edges(data=True):
            d.pop('valence', None)


if __name__ == '__main__':
    from utils import load_data
    data = load_data('aids700nef', False)
    for g in data.graphs:
        print(g.graph['gid'])
