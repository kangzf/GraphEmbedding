from utils import get_data, draw_graph, get_root_path
from distance import GED
import networkx as nx
from time import time
from random import randint
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib


def exp1():
    g0 = create_graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5)])
    g1 = create_graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)])
    # draw_graph(g0, get_root_path() + '/files/exp_g0.png')
    # draw_graph(g1, get_root_path() + '/files/exp_g1.png')
    print(GED(g0, g1))
    nx.set_node_attributes(g0, 'label', {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    nx.set_node_attributes(g1, 'label', {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1})
    print(GED(g0, g1))

def exp2():
    g0 = nx.Graph()
    g0.add_node(0)
    g1 = create_graph([(0, 1)])
    nx.set_node_attributes(g0, 'label', {0: 0})
    nx.set_node_attributes(g1, 'label', {0: 0, 1: 1})
    print(GED(g0, g1))


def exp3():
    g0 = create_graph([(0, 1), (1, 2), (2, 0)])
    g1 = create_graph([(0, 1)])
    nx.set_node_attributes(g0, 'label', {0: 1, 1: 1, 2: 0})
    nx.set_node_attributes(g1, 'label', {0: 1, 1: 0})
    print(GED(g0, g1))

def exp4():
    file = open(get_root_path() + '/files/ged_LSAP_2.csv', 'w')
    xs = list(range(10, 140, 10))
    ys = list(range(10, 140, 10))
    cnt = 10
    def print_and_log(s, file):
        print(s)
        file.write(s + '\n')
        file.flush()
    print_and_log('g1_node,g2_node,g1_edge,g2_edge,ged,time_sec', file)
    for x in xs:
        for y in ys:
            for i in range(cnt):
                def generate_random_graph(n):
                    return nx.gnm_random_graph(n, randint(0, n * (n - 1) / 2))
                g1 = generate_random_graph(x)
                g2 = generate_random_graph(y)
                t = time()
                d = GED(g1, g2)
                s = '{},{},{},{},{},{:.5f}'.format( \
                    g1.number_of_nodes(), g2.number_of_nodes(), \
                    g1.number_of_edges(), g2.number_of_edges(), \
                    d, time() - t)
                print_and_log(s, file)
    file.close()


def exp5():
    font = {'family': 'serif',
            'size': 22}
    matplotlib.rc('font', **font)
    file = 'ged_LSAP'
    data = read_csv(get_root_path() + '/files/{}.csv'.format(file))
    print(data)
    plt.scatter(data['g2_node'], data['time_sec'], label="LSAP (ged4py)")
    plt.xlabel('# nodes of graph 2')
    plt.ylabel('time (sec)')
    plt.legend(loc='best')
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    plt.savefig(get_root_path() + '/files/{}.png'.format(file))
    plt.show()

def create_graph(edges):
    g = nx.Graph()
    for edge in edges:
        g.add_edge(*edge)
    return g

exp5()