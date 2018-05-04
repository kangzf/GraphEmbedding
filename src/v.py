from utils import get_root_path
import networkx as nx


def vis(q, gs, addi_str=None):
    print('TODO')


def t():
    dir = get_root_path() + '/data/AIDS/'
    vis(load_graph(dir, 0), [load_graph(dir, 10000),
                             load_graph(dir, 20000),
                             load_graph(dir, 30000),
                             load_graph(dir, 40000)],
        ['ged=5', 'ged=6\ntime=0.02sec', 'xxx', ''])


def load_graph(dir, gid):
    return nx.read_gexf(dir + '{}.gexf'.format(gid))


if __name__ == '__main__':
    t()
