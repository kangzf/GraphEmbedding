from utils import get_root_path
import networkx as nx


def vis(q, gs, info_dict=None):
    print('TODO')


def t():
    dir = get_root_path() + '/data/AIDS/'
    d = {'ged': [5, 4, 3, 2], 'time': ['0.01sec', '', '', '1.1sec']}
    vis(load_graph(dir, 0), [load_graph(dir, 10000),
                             load_graph(dir, 20000),
                             load_graph(dir, 30000),
                             load_graph(dir, 40000)], d)


def load_graph(dir, gid):
    return nx.read_gexf(dir + '{}.gexf'.format(gid))


if __name__ == '__main__':
    t()
