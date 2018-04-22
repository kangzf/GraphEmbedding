from utils import get_data, draw_graph, get_root_path
from distance import GED
import networkx as nx


def exp1():
    g0 = create_graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5)])
    g1 = create_graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)])
    # draw_graph(g0, get_root_path() + '/files/exp_g0.png')
    # draw_graph(g1, get_root_path() + '/files/exp_g1.png')
    print(GED(g0, g1))
    nx.set_node_attributes(g0, 'label', {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 1})
    nx.set_node_attributes(g1, 'label', {0: 1, 0: 1, 0: 0, 2: 0, 3: 0, 5: 1})
    print(GED(g0, g1))



def create_graph(edges):
    g = nx.Graph()
    for edge in edges:
        g.add_edge(*edge)
    return g

exp1()