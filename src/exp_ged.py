from utils import get_root_path, get_data, get_ts
from distance import astar_ged, beam_ged, hungarian_ged, vj_ged, ged
import networkx as nx
from time import time
from random import randint, uniform
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from glob import glob

args1 = {'astar': {'color': 'grey'},
         'beam5': {'color': 'red'},
         'beam10': {'color': 'b'},
         'beam20': {'color': 'forestgreen'},
         'beam40': {'color': 'darkorange'},
         'beam80': {'color': 'cyan'},
         'hungarian': {'color': 'deepskyblue'},
         'vj': {'color': 'red'}}
args2 = {'astar': {'marker': '*', 'facecolors': 'none', 'edgecolors': 'grey'},
         'beam5': {'marker': '|', 'facecolors': 'red'},
         'beam10': {'marker': '_', 'facecolors': 'b'},
         'beam20': {'marker': 'D', 'facecolors': 'none',
                    'edgecolors': 'forestgreen'},
         'beam40': {'marker': '^', 'facecolors': 'none',
                    'edgecolors': 'darkorange'},
         'beam80': {'marker': 's', 'facecolors': 'none', 'edgecolors': 'cyan'},
         'hungarian': {'marker': 'X', 'facecolors': 'none',
                       'edgecolors': 'deepskyblue'},
         'vj': {'marker': 'P', 'facecolors': 'none', 'edgecolors': 'red'}}


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


def create_graph(edges):
    g = nx.Graph()
    for edge in edges:
        g.add_edge(*edge)
    return g


def exp4():
    ms = ['astar', 'beam5', 'beam10', 'beam20', 'beam40', 'beam80',
          'hungarian', 'vj']
    fn = '_'.join(ms)
    file = open(get_root_path() + '/files/ged_{}_{}.csv'.format(fn, get_ts()),
                'w')
    xs = [10]
    ys = list(range(10, 141, 10))
    cnt = 10
    ged_s = ','.join(['ged_' + i for i in ms])
    time_s = ','.join(['time_' + i for i in ms])
    print_and_log('g1_node,g2_node,g1_edge,g2_edge,{},{}'.format( \
        ged_s, time_s), file)
    for x in xs:
        for y in ys:
            for i in range(cnt):
                g1 = generate_random_graph(x)
                g2 = generate_random_graph(y)
                ds = []
                ts = []
                for m in ms:
                    t = time()
                    d = ged(g1, g2, m)
                    t = time() - t
                    ds.append(d)
                    ts.append(t)
                s = '{},{},{},{},{},{}'.format( \
                    g1.number_of_nodes(), g2.number_of_nodes(), \
                    g1.number_of_edges(), g2.number_of_edges(), \
                    ','.join(str(i) for i in ds),
                    ','.join(['{:.5f}'.format(i) for i in ts]))
                print_and_log(s, file)
                # if d1 < 0:
                #     exit(-1)
    file.close()


def print_and_log(s, file):
    print(s)
    file.write(s + '\n')
    file.flush()


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
    file = 'ged_astar_beam5_beam10_beam20_beam40_beam80_hungarian_vj_2018-04-29T14:56:38.676491'
    data = read_csv(get_root_path() + '/files/{}.csv'.format(file))
    models = []
    for model in data.columns.values:
        if 'time' in model:
            models.append(model.split('_')[1])
    if 'astar' in models:
        for i, t in data['time_astar'].iteritems():
            if t >= 300:
                data.loc[i, 'time_astar'] = 300
                data.loc[i, 'ged_astar'] = 0
    print(data)
    plt.figure(0)
    plt.figure(figsize=(16, 10))
    for i, model in enumerate(models):
        if model == 'astar':
            continue
        plt.scatter(data['g2_node'], data['time_' + model], s=150, label=model,
                    **args2[model])
    plt.xlabel('# nodes of graph 2')
    plt.ylabel('time (sec)')
    plt.legend(loc='best')
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    plt.savefig(get_root_path() + '/files/{}_time.png'.format(file))
    # plt.show()
    plt.figure(1)
    plt.figure(figsize=(11, 11))
    for i, model in enumerate(models):
        plt.scatter(data['ged_astar'], data['ged_' + model], s=150, label=model,
                    **args[i])
    plt.xlabel('true ged')
    plt.ylabel('ged')
    plt.xlim(1, 57)
    plt.legend(loc='best')
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    plt.savefig(get_root_path() + '/files/{}_ged.png'.format(file))
    # plt.show()


def exp6():
    g0 = nx.Graph()
    g0.add_node(0)
    g1 = create_graph([(0, 1), (0, 2), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
    print(hungarian_ged(g0, g1))


def exp7():
    dataset = 'aids10k'
    model = 'vj'
    train_data = get_data(dataset, True)
    test_data = get_data(dataset, False)
    m = len(test_data.graphs)
    n = len(train_data.graphs)
    ged_mat = np.zeros((m, n))
    time_mat = np.zeros((n, n))
    outdir = get_root_path() + '/files'
    file = open('{}/ged_{}_{}_{}.csv'.format( \
        outdir, dataset, model, get_ts()), 'w')
    print_and_log('i,j,i_node,j_node,i_edge,j_edge,ged,time', file)
    for i in range(m):
        for j in range(n):
            g1 = test_data.graphs[i]
            g2 = train_data.graphs[j]
            t = time()
            d = ged(g1, g2, model)
            t = time() - t
            s = '{},{},{},{},{},{},{},{:.5f}'.format(i, j, \
                                                     g1.number_of_nodes(),
                                                     g2.number_of_nodes(), \
                                                     g1.number_of_edges(),
                                                     g2.number_of_edges(), \
                                                     d, t)
            print_and_log(s, file)
            ged_mat[i][j] = d
            time_mat[i][j] = t
    file.close()
    np.save('{}/ged_ged_mat_{}_{}_{}'.format( \
        outdir, dataset, model, get_ts()), ged_mat)
    np.save('{}/ged_time_mat_{}_{}_{}'.format( \
        outdir, dataset, model, get_ts()), time_mat)


class Metric(object):
    def __init__(self, name, ylabel):
        self.name = name
        self.ylabel = ylabel

    def __str__(self):
        return self.name


def exp8():
    dataset = 'aids10k'
    models = ['astar', 'beam5', 'beam10', 'beam20', 'beam40', 'beam80', \
              'hungarian', 'vj']
    metrics = [Metric('ged', 'ged'), Metric('time', 'time (sec)')]
    for metric in metrics:
        exp8_helper(dataset, models, metric)


def exp8_helper(dataset, models, metric):
    font = {'family': 'serif',
            'size': 22}
    matplotlib.rc('font', **font)

    plt.figure(0)
    plt.figure(figsize=(16, 10))

    xs = get_test_graph_sizes(dataset)
    so = np.argsort(xs)
    xs.sort()
    for model in models:
        if model == 'astar':
            continue
        mat = get_result_mat(metric, dataset, model)
        print('plotting for {}'.format(model))
        ys = np.mean(mat, 1)[so]
        plt.plot(xs, ys, **args1[model])
        plt.scatter(xs, ys, s=200, label=model, **args2[model])
    plt.xlabel('query graph size')
    ax = plt.gca()
    ax.set_xticks(xs)
    plt.ylabel('average {}'.format(metric.ylabel))
    plt.legend(loc='best', ncol=2)
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    plt.savefig(get_root_path() + '/files/ged_{}_mat_{}_{}.png'.format( \
        metric, dataset, '_'.join(models)))


def get_result_mat(metric, dataset, model):
    file_p = get_root_path() + '/files/ged_{}_mat_{}_{}_*.npy'.format( \
        metric, dataset, model)
    li = glob(file_p)
    if len(li) != 1:
        raise RuntimeError('Files for {}: {}'.format(file_p, li))
    file = li[0]
    return np.load(file)


def get_test_graph_sizes(dataset):
    test_data = get_data(dataset, train=False)
    return [test_data.graphs[i].number_of_nodes() for i in \
            range(len(test_data.gids))]


def exp9():
    dataset = 'aids10k'
    models = ['beam5', 'beam10', 'beam20', 'beam40', 'beam80', \
              'hungarian', 'vj']
    true_mat = get_result_mat('ged', dataset, 'beam80')
    ks = []
    k = 1
    while k < true_mat.shape[1]:
        ks.append(k)
        k *= 2
    # print_ids = range(true_mat.shape[0])
    print_ids = []

    font = {'family': 'serif',
            'size': 22}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(16, 10))

    for model in models:
        print(model)
        pred_mat = get_result_mat('ged', dataset, model)
        aps = precision_at_ks(true_mat, pred_mat, ks, print_ids)
        #print('aps {}: {}'.format(model, aps))
        plt.semilogx(ks, aps, **args1[model])
        plt.scatter(ks, aps, s=200, label=model, **args2[model])
    plt.xlabel('k')
    # ax = plt.gca()
    # ax.set_xticks(ks)
    plt.ylabel('ap@k')
    plt.legend(loc='best', ncol=2)
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    plt.savefig(get_root_path() + '/files/ged_{}_{}_{}.png'.format( \
      'apk', dataset, '_'.join(models)))


def precision_at_ks(true_mat, pred_mat, ks, print_ids=[]):
    x = np.argsort(true_mat)
    # print(x)
    y = np.argsort(pred_mat)
    # print(y)
    m, n = true_mat.shape
    assert (true_mat.shape == pred_mat.shape)
    ps = np.zeros((m, len(ks)))
    for i in range(m):
        for k_idx, k in enumerate(ks):
            assert (type(k) is int and k > 0 and k < n)
            ps[i][k_idx] = len(set(x[i][:k]).intersection(y[i][:k])) / k
        if i in print_ids:
            print('query {}\nks:    {}\nprecs: {}'.format(i, ks, ps[i]))
    return np.mean(ps, axis=0)


exp9()
