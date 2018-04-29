from utils import get_root_path
from distance import hungarian_ged, astar_ged
import networkx as nx
from time import time
from random import randint, uniform
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib


def exp1():
    g0 = create_graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5)])
    g1 = create_graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)])
    # draw_graph(g0, get_root_path() + '/files/exp_g0.png')
    # draw_graph(g1, get_root_path() + '/files/exp_g1.png')
    print('hungarian_ged', hungarian_ged(g0, g1))
    print('astar_ged', astar_ged(g0, g1))
    nx.set_node_attributes(g0, 'label', {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    nx.set_node_attributes(g1, 'label', {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1})
    print('hungarian_ged', hungarian_ged(g0, g1))
    print('astar_ged', astar_ged(g0, g1))

def exp2():
    g0 = nx.Graph()
    g0.add_node(0)
    g1 = create_graph([(0, 1)])
    nx.set_node_attributes(g0, 'label', {0: 0})
    nx.set_node_attributes(g1, 'label', {0: 0, 1: 1})
    print('hungarian_ged', hungarian_ged(g0, g1))
    print('astar_ged', astar_ged(g0, g1))


def exp3():
    g0 = create_graph([(0, 1), (1, 2), (2, 0)])
    g1 = create_graph([(0, 1)])
    nx.set_node_attributes(g0, 'label', {0: 1, 1: 1, 2: 0})
    nx.set_node_attributes(g1, 'label', {0: 1, 1: 0})
    print(hungarian_ged(g0, g1))

def exp4():
    file = open(get_root_path() + '/files/ged_astar_lsap.csv', 'w')
    xs = [10]
    ys = list(range(1, 141, 1))
    cnt = 10
    def print_and_log(s, file):
        print(s)
        file.write(s + '\n')
        file.flush()
    print_and_log('g1_node,g2_node,g1_edge,g2_edge,ged_astar,ged_lsap,'
                  'time_sec_astar,time_sec_lsap', file)
    for x in xs:
        for y in ys:
            for i in range(cnt):
                g1 = generate_random_graph(x)
                g2 = generate_random_graph(y)
                t = time()
                d1 = astar_ged(g1, g2)
                t1 = time() - t
                t = time()
                d2 = hungarian_ged(g1, g2)
                t2 = time() - t
                s = '{},{},{},{},{},{},{:.5f},{:.5f}'.format( \
                    g1.number_of_nodes(), g2.number_of_nodes(), \
                    g1.number_of_edges(), g2.number_of_edges(), \
                    d1, d2, t1, t2)
                print_and_log(s, file)
                if d1 < 0:
                    exit(-1)
    file.close()


def generate_random_graph(n, connected=True):
    if connected:
        while True:
            g = nx.erdos_renyi_graph(n, uniform(0, 1))
            if nx.is_connected(g):
                break
    else:
        g = nx.gnm_random_graph(n, randint(0, n * (n - 1) / 2))
    return g


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

def exp6():
    g0 = nx.Graph()
    g0.add_node(0)
    g1 = create_graph([(0, 1), (0, 2), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
    print(hungarian_ged(g0, g1))

def create_graph(edges):
    g = nx.Graph()
    for edge in edges:
        g.add_edge(*edge)
    return g


exp4()
