import sys

sys.path.append('..')
from utils import get_root_path, get_data_path, exec, get_file_base_id, \
    load_data, \
    save_as_dict, load_as_dict, get_save_path
import networkx as nx
from glob import glob
from collections import defaultdict
import random
from random import sample, shuffle

random.seed(123)


def gen_graphs():
    # dirin = get_data_path() + '/Linux'
    # file = dirin + '/Linux.txt'
    # train_dirout = dirin + '/train'
    # test_dirout = dirin + '/test'
    dirin = get_data_path() + '/iGraph20/datasets'
    file = dirin + '/nasa.igraph'
    train_dirout = dirin + '/train'
    test_dirout = dirin + '/test'
    graphs = {}
    gid = None
    types = set()
    disconnects = set()
    with open(file) as f:
        for line in f:
            ls = line.rstrip().split()
            if ls[0] == 't':
                assert (len(ls) == 3)
                assert (ls[1] == '#')
                if gid:
                    assert (gid not in graphs)
                    graphs[gid] = g
                    if not nx.is_connected(g):
                        disconnects.add(g)
                g = nx.Graph()
                gid = int(ls[2])
                print(gid)
            elif ls[0] == 'v':
                assert (len(ls) == 3)
                type = int(ls[2])
                types.add(type)
                g.add_node(int(ls[1]), type=type)
            elif ls[0] == 'e':
                assert (len(ls) == 4)
                edge_type = int(ls[3])
                assert(edge_type == 0)
                g.add_edge(int(ls[1]), int(ls[2]))

    print(len(types), 'node types')
    print(len(disconnects), 'disconnected graphs')
    print(types)

def vis_example():
    file =  get_data_path() + '/xxx.graphml'
    g = nx.read_graphml(file)
    print(g)

gen_graphs()
