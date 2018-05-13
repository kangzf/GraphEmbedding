from utils import get_root_path, load_data, get_ts
from distance import astar_ged, beam_ged, hungarian_ged, vj_ged, ged
from result import load_results_as_dict, load_result
from vis import vis
import networkx as nx
from time import time
from random import randint, uniform
from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from glob import glob

args1 = {'astar': {'color': 'grey'},
         'beam5': {'color': 'deeppink'},
         'beam10': {'color': 'b'},
         'beam20': {'color': 'forestgreen'},
         'beam40': {'color': 'darkorange'},
         'beam80': {'color': 'cyan'},
         'hungarian': {'color': 'deepskyblue'},
         'vj': {'color': 'red'},
         'graph2vec': {'color': 'darkcyan'}}
args2 = {'astar': {'marker': '*', 'facecolors': 'none', 'edgecolors': 'grey'},
         'beam5': {'marker': '|', 'facecolors': 'deeppink'},
         'beam10': {'marker': '_', 'facecolors': 'b'},
         'beam20': {'marker': 'D', 'facecolors': 'none',
                    'edgecolors': 'forestgreen'},
         'beam40': {'marker': '^', 'facecolors': 'none',
                    'edgecolors': 'darkorange'},
         'beam80': {'marker': 's', 'facecolors': 'none', 'edgecolors': 'cyan'},
         'hungarian': {'marker': 'X', 'facecolors': 'none',
                       'edgecolors': 'deepskyblue'},
         'vj': {'marker': 'P', 'facecolors': 'none', 'edgecolors': 'red'},
         'graph2vec': {'marker': 'h', 'facecolors': 'none', 'edgecolors': 'darkcyan'}}


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
    train_data = load_data(dataset, True)
    test_data = load_data(dataset, False)
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
    # Plot ged and time.
    dataset = 'aids10k'
    models = ['beam5', 'beam10', 'beam20', 'beam40', 'beam80', \
              'hungarian', 'vj']
    rs = load_results_as_dict(dataset, models)
    metrics = [Metric('ged', 'ged'), Metric('time', 'time (sec)')]
    for metric in metrics:
        exp8_helper(dataset, models, metric, rs)


def exp8_helper(dataset, models, metric, rs):
    font = {'family': 'serif',
            'size': 22}
    matplotlib.rc('font', **font)

    plt.figure(0)
    plt.figure(figsize=(16, 10))

    xs = get_test_graph_sizes(dataset)
    so = np.argsort(xs)
    xs.sort()
    for model in models:
        mat = rs[model].mat(metric.name)
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
    plt.savefig(get_root_path() + '/files/{}/{}/ged_{}_mat_{}_{}.png'.format( \
        dataset, metric, metric, dataset, '_'.join(models)))


def get_test_graph_sizes(dataset):
    test_data = load_data(dataset, train=False)
    return [g.number_of_nodes() for g in test_data.graphs]


def exp9():
    # Plot ap@k.
    dataset = 'aids10k'
    models = ['beam5', 'beam10', 'beam20', 'beam40', 'beam80', \
              'hungarian', 'vj', 'graph2vec']
    true_model = 'beam80'
    metric = 'ap@k'
    rs = load_results_as_dict(dataset, models)
    true_result = rs[true_model]

    ks = []
    k = 1
    while k < true_result.ged_mat().shape[1]:
        ks.append(k)
        k *= 2
    exp9_helper(dataset, models, rs, true_result, metric, ks, True)

    ks = range(1, 31)
    exp9_helper(dataset, models, rs, true_result, metric, ks, False)

def exp9_helper(dataset, models, rs, true_result, metric, ks, logscale):
    # print_ids = range(true_mat.shape[0])
    print_ids = []
    font = {'family': 'serif',
            'size': 22}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(16, 10))
    for model in models:
        print(model)
        aps = precision_at_ks(true_result, rs[model], ks, print_ids)
        # print('aps {}: {}'.format(model, aps))
        if logscale:
            pltfunc = plt.semilogx
        else:
            pltfunc = plt.plot
        pltfunc(ks, aps, **args1[model])
        plt.scatter(ks, aps, s=200, label=model, **args2[model])
    plt.xlabel('k')
    # ax = plt.gca()
    # ax.set_xticks(ks)
    plt.ylabel(metric)
    plt.legend(loc='best', ncol=2)
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    kss = 'k_{}_{}'.format(min(ks), max(ks))
    plt.savefig(get_root_path() + '/files/{}/{}/ged_{}_{}_{}_{}.png'.format( \
        dataset, metric, metric, dataset, '_'.join(models), kss))


def precision_at_ks(true_r, pred_r, ks, print_ids=[]):
    true_ids = true_r.ged_sort_id_mat()
    # print(x)
    pred_ids = pred_r.ged_sort_id_mat()
    # print(y)
    m, n = true_ids.shape
    assert (true_ids.shape == pred_ids.shape)
    ps = np.zeros((m, len(ks)))
    for i in range(m):
        for k_idx, k in enumerate(ks):
            assert (type(k) is int and k > 0 and k < n)
            ps[i][k_idx] = len(set(true_ids[i][:k]).intersection( \
                pred_ids[i][:k])) / k
        if i in print_ids:
            print('query {}\nks:    {}\nprecs: {}'.format(i, ks, ps[i]))
    return np.mean(ps, axis=0)


def exp10():
    # Query visualization.
    dataset = 'aids10k'
    model = 'graph2vec'
    k = 5
    info_dict = {
        # draw node config
        'draw_node_size': 10,
        'draw_node_label_enable': True,
        'node_label_name': 'type',
        'draw_node_label_font_size': 8,
        'draw_node_color_map': {'C': 'red',
                                'O': 'blue',
                                'N': 'green'},
        # draw edge config
        'draw_edge_label_enable': True,
        'edge_label_name': 'valence',
        'draw_edge_label_font_size': 6,
        # graph text info config
        'each_graph_text_list': [],
        'each_graph_text_font_size': 8,
        'each_graph_text_pos': [0.5, 1.05],
        # graph padding: value range: [0, 1]
        'top_space': 0.25,  # out of whole graph
        'bottom_space': 0,
        'hbetween_space': 1,  # out of the subgraph
        'wbetween_space': 0.02,
        # plot config
        'each_graph_text_pos': [0.5, 0.8],
        'each_graph_font_size': 10,
        'plot_dpi': 200,
        'plot_save_path': ''
    }
    r = load_result(dataset, model)
    ged_mat = r.ged_mat()
    time_mat = r.time_mat()
    ids = r.ged_sort_id_mat()
    m, n = ged_mat.shape
    train_data = load_data(dataset, train=True)
    test_data = load_data(dataset, train=False)
    for i in range(m):
        q = test_data.graphs[i]
        gids = ids[i][:k]
        gs = [train_data.graphs[j] for j in gids]
        info_dict['each_graph_text_list'] = \
            [get_text_label(ged_mat, time_mat, i, i, q, model, True)] + \
            [get_text_label(ged_mat, time_mat, i, j, \
                            train_data.graphs[j], model, False) for j in gids]
        info_dict['plot_save_path'] = \
            get_root_path() + \
            '/files/{}/query_vis/{}/query_vis_{}_{}_{}.png'.format( \
                dataset, model, dataset, model, i)
        vis(q, gs, info_dict)


def get_text_label(ged_mat, time_mat, i, j, g, model, is_query):
    rtn = '\n\nid: {}\norig id: {}{}'.format( \
        j, g.graph['gid'], get_graph_stats_text(g))
    if is_query:
        rtn += '\nquery\nmodel: {}'.format(model)
    else:
        rtn += '\n ged: {}\ntime: {:.2f} sec'.format( \
            ged_mat[i][j], time_mat[i][j])
    return rtn

def get_graph_stats_text(g):
    return '\n#nodes: {}\n#edges: {}\ndensity: {:.2f}'.format( \
        g.number_of_nodes(), g.number_of_edges(), nx.density(g))



exp10()
