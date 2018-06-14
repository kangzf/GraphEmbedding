#!/usr/bin/env python3
from utils import get_result_path, load_data, get_ts, \
    exec_turnoff_print, get_computer_name, check_nx_version, prompt_get_cpu
from metrics import Metric, precision_at_ks, mean_reciprocal_rank
from distance import ged
from result import load_results_as_dict, load_result
import networkx as nx

check_nx_version()
import multiprocessing as mp
from time import time
from random import randint, uniform

if get_computer_name() == 'yba':  # local
    from pandas import read_csv
    import matplotlib.pyplot as plt
    import matplotlib
    from vis import vis
import numpy as np

""" Plotting args. """
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
         'graph2vec': {'marker': 'h', 'facecolors': 'none',
                       'edgecolors': 'darkcyan'}}


def exp1():
    """ Toy. """
    g0 = create_graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5)])
    g1 = create_graph([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)])
    # draw_graph(g0, get_result_path() + '/syn/exp1_g0.png')
    # draw_graph(g1, get_result_path() + '/syn/exp1_g1.png')
    print('hungarian_ged', ged(g0, g1, 'hungarian'))
    print('astar_ged', ged(g0, g1, 'astar'))
    nx.set_node_attributes(g0, 'label', {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0})
    nx.set_node_attributes(g1, 'label', {0: 1, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1})
    print('hungarian_ged', ged(g0, g1, 'hungarian'))
    print('astar_ged', ged(g0, g1, 'astar'))
    # Another toy.
    g0 = nx.Graph()
    g0.add_node(0)
    g1 = create_graph([(0, 1)])
    nx.set_node_attributes(g0, 'label', {0: 0})
    nx.set_node_attributes(g1, 'label', {0: 0, 1: 1})
    print('hungarian_ged', ged(g0, g1, 'hungarian'))
    print('astar_ged', ged(g0, g1, 'astar'))
    # Another toy.
    g0 = create_graph([(0, 1), (1, 2), (2, 0)])
    g1 = create_graph([(0, 1)])
    nx.set_node_attributes(g0, 'label', {0: 1, 1: 1, 2: 0})
    nx.set_node_attributes(g1, 'label', {0: 1, 1: 0})
    print(ged(g0, g1, 'hungarian'))
    # Another toy.
    g0 = nx.Graph()
    g0.add_node(0)
    g1 = create_graph([(0, 1), (0, 2), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)])
    print(ged(g0, g1, 'hungarian'))


def create_graph(edges):
    g = nx.Graph()
    for edge in edges:
        g.add_edge(*edge)
    return g


def exp2():
    """ Run baselines on a synthetic dataset. """
    ms = ['astar', 'beam5', 'beam10', 'beam20', 'beam40', 'beam80',
          'hungarian', 'vj']
    fn = '_'.join(ms)
    file = open(get_result_path() + '/ged_{}_{}.csv'.format(fn, get_ts()),
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


def exp3():
    """ Plot ged and time for the synthetic dataset. """
    font = {'family': 'serif',
            'size': 22}
    matplotlib.rc('font', **font)
    file = 'ged_astar_beam5_beam10_beam20_beam40_beam80_hungarian_vj_2018-04-29T14:56:38.676491'
    data = read_csv(get_result_path() + '/syn/{}.csv'.format(file))
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
    plt.savefig(get_result_path() + '/syn/{}_time.png'.format(file))
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
    plt.savefig(get_result_path() + '/syn/{}_ged.png'.format(file))
    # plt.show()


def exp4():
    """ Run baselines on real datasets. Take a while. """
    dataset = 'aids50'
    model = 'beam80'
    row_graphs = load_data(dataset, train=False)
    col_graphs = load_data(dataset, train=True)
    num_cpu = prompt_get_cpu()
    exec_turnoff_print()
    real_dataset_run_helper(dataset, model, row_graphs, col_graphs, num_cpu)


def real_dataset_run_helper(dataset, model, row_graphs, col_graphs, num_cpu):
    m = len(row_graphs.graphs)
    n = len(col_graphs.graphs)
    ged_mat = np.zeros((m, n))
    time_mat = np.zeros((m, n))
    outdir = '{}/{}'.format(get_result_path(), dataset)
    computer_name = get_computer_name()
    csv_fn = '{}/csv/ged_{}_{}_{}_{}_{}cpus.csv'.format( \
        outdir, dataset, model, get_ts(), computer_name, num_cpu)
    file = open(csv_fn, 'w')
    print('Saving to {}'.format(csv_fn))
    print_and_log('i,j,i_gid,j_gid,i_node,j_node,i_edge,j_edge,ged,lcnt,time(msec)', file)
    # Multiprocessing.
    pool = mp.Pool(processes=num_cpu)
    # print('Using {} CPUs'.format(cpu_count()))
    # Submit to pool workers.
    results = [[None] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            g1 = row_graphs.graphs[i]
            g2 = col_graphs.graphs[j]
            results[i][j] = pool.apply_async( \
                ged, args=(g1, g2, model, True, True,))
        print_progress(i, j, m, n, 'submit: {} {} {} cpus;'. \
                       format(model, computer_name, num_cpu))
    # Retrieve results from pool workers.
    for i in range(m):
        for j in range(n):
            print_progress(i, j, m, n, 'work: {} {} {} cpus;'. \
                           format(model, computer_name, num_cpu))
            d, lcnt, g1_a, g2_a, t = results[i][j].get()
            g1 = row_graphs.graphs[i]
            g2 = col_graphs.graphs[j]
            assert (g1.number_of_nodes() == g1_a.number_of_nodes())
            assert (g2.number_of_nodes() == g2_a.number_of_nodes())
            s = '{},{},{},{},{},{},{},{},{},{},{:.2f}'.format( \
                i, j, g1.graph['gid'], g2.graph['gid'], \
                g1.number_of_nodes(), g2.number_of_nodes(), \
                g1.number_of_edges(), g2.number_of_edges(), \
                d, lcnt, t)
            print_and_log(s, file)
            ged_mat[i][j] = d
            time_mat[i][j] = t
    file.close()
    np.save('{}/ged/ged_ged_mat_{}_{}_{}_{}_{}cpus'.format( \
        outdir, dataset, model, get_ts(), computer_name, num_cpu), ged_mat)
    np.save('{}/time/ged_time_mat_{}_{}_{}_{}_{}cpus'.format( \
        outdir, dataset, model, get_ts(), computer_name, num_cpu), time_mat)


def print_progress(i, j, m, n, label):
    cur = i * n + j
    tot = m * n
    print('----- {} progress: {}/{}={:.1%}'.format(label, cur, tot, cur / tot))


def exp5():
    """ Plot ged and time. """
    dataset = 'aids50'
    models = ['beam5', 'beam10', 'beam20', 'beam40', 'beam80', \
              'hungarian', 'vj']
    rs = load_results_as_dict(dataset, models)
    metrics = [Metric('ged', 'ged'), Metric('time', 'time (msec)')]
    for metric in metrics:
        plot_ged_time_helper(dataset, models, metric, rs)


def plot_ged_time_helper(dataset, models, metric, rs):
    font = {'family': 'serif',
            'size': 22}
    matplotlib.rc('font', **font)

    plt.figure(0)
    plt.figure(figsize=(16, 10))

    xs = get_test_graph_sizes(dataset)
    so = np.argsort(xs)
    xs.sort()
    for model in models:
        mat = rs[model].mat(metric.name, norm=True)
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
    sp = get_result_path() + '/{}/{}/ged_{}_mat_{}_{}.png'.format( \
        dataset, metric, metric, dataset, '_'.join(models))
    plt.savefig(sp)
    print('Saved to {}'.format(sp))


def get_test_graph_sizes(dataset):
    test_data = load_data(dataset, train=False)
    return [g.number_of_nodes() for g in test_data.graphs]


def exp6():
    """ Plot ap@k. """
    dataset = 'aids50'
    models = ['beam5', 'beam10', 'beam20', 'beam40', 'beam80', \
              'hungarian', 'vj']
    true_model = 'beam80'
    metric = 'ap@k'
    norms = [True, False]
    rs = load_results_as_dict(dataset, models)
    true_result = rs[true_model]
    for norm in norms:
        ks = []
        k = 1
        _, n = true_result.m_n()
        while k < n:
            ks.append(k)
            k *= 2
        plot_apk_helper(dataset, models, rs, true_result, metric, norm, ks, True)
        ks = range(1, 31)
        plot_apk_helper(dataset, models, rs, true_result, metric, norm, ks, False)


def plot_apk_helper(dataset, models, rs, true_result, metric, norm, ks, logscale):
    print_ids = []
    font = {'family': 'serif',
            'size': 22}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(16, 10))
    for model in models:
        print(model)
        aps = precision_at_ks(true_result, rs[model], norm, ks, print_ids)
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
    plt.ylim([-0.06, 1.06])
    plt.legend(loc='best', ncol=2)
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    kss = 'k_{}_{}'.format(min(ks), max(ks))
    sp = get_result_path() + '/{}/{}/ged_{}_{}_{}_{}_{}.png'.format( \
        dataset, metric, metric, dataset, '_'.join(models), kss,
        get_norm_str(norm))
    plt.savefig(sp)
    print('Saved to {}'.format(sp))


def get_norm_str(norm):
    if norm:
        return 'norm'
    else:
        return 'nonorm'


def exp7():
    """ Plot mrr. """
    dataset = 'aids50'
    models = ['beam5', 'beam10', 'beam20', 'beam40', 'beam80', \
              'hungarian', 'vj']
    true_model = 'beam80'
    metric = 'mrr'
    norms = [True, False]
    rs = load_results_as_dict(dataset, models)
    true_result = rs[true_model]
    for norm in norms:
        plot_mrr_helper(dataset, models, rs, true_result, metric, norm)


def plot_mrr_helper(dataset, models, rs, true_result, metric, norm):
    print_ids = []
    font = {'family': 'serif',
            'size': 22}
    matplotlib.rc('font', **font)
    plt.figure(figsize=(16, 10))
    mrrs = []
    for model in models:
        print(model)
        mrr = mean_reciprocal_rank(true_result, rs[model], norm, print_ids)
        print('mrr {}: {}'.format(model, mrr))
        mrrs.append(mrr)
    ind = np.arange(len(mrrs))  # the x locations for the groups
    width = 0.35  # the width of the bars
    bars = plt.bar(ind, mrrs, width)
    for i, bar in enumerate(bars):
        bar.set_color(args1[models[i]]['color'])
    autolabel(bars)
    plt.xlabel('model')
    plt.xticks(ind, models)
    plt.ylabel(metric)
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    sp = get_result_path() + '/{}/{}/ged_{}_{}_{}_{}.png'.format( \
        dataset, metric, metric, dataset, '_'.join(models),
        get_norm_str(norm))
    plt.savefig(sp)
    print('Saved to {}'.format(sp))


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text( \
            rect.get_x() + rect.get_width()/2., 1.005*height, \
            '{:.2f}'.format(height), ha='center', va='bottom')


def exp8():
    """ Query visualization. """
    dataset = 'aids10k'
    model = 'graph2vec'
    true_model = 'beam80'
    norms = [True, False]
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
        'plot_dpi': 200,
        'plot_save_path': ''
    }
    r = load_result(dataset, model)
    tr = load_result(dataset, true_model)
    for norm in norms:
        ids = r.sort_id_mat(norm)
        m, n = r.m_n()
        train_data = load_data(dataset, train=True)
        test_data = load_data(dataset, train=False)
        for i in range(m):
            q = test_data.graphs[i]
            gids = ids[i][:k]
            gs = [train_data.graphs[j] for j in gids]
            info_dict['each_graph_text_list'] = \
                [get_text_label(r, tr, i, i, q, model, norm, True)] + \
                [get_text_label(r, tr, i, j, \
                                train_data.graphs[j], model, norm, False) \
                 for j in gids]
            info_dict['plot_save_path'] = \
                get_result_path() + \
                '/{}/query_vis/{}/query_vis_{}_{}_{}_{}.png'.format( \
                    dataset, model, dataset, model, i, get_norm_str(norm))
            vis(q, gs, info_dict)


def get_text_label(r, tr, qid, gid, g, model, norm, is_query):
    if is_query or r.model_ == tr.model_:
        rtn = '\n\n'
    else:
        ged_str = get_ged_select_norm_str(tr, qid, gid, norm)
        rtn = 'true ged: {}\ntrue rank: {}\n'.format( \
            ged_str, tr.ranking(qid, gid, norm))
    rtn += 'id: {}\norig id: {}{}'.format( \
        gid, g.graph['gid'], get_graph_stats_text(g))
    if is_query:
        rtn += '\nquery\nmodel: {}'.format(model)
    else:
        ged_sim_str, ged_sim = r.dist_sim(qid, gid, norm)
        if ged_sim_str == 'ged':
            ged_str = get_ged_select_norm_str(r, qid, gid, norm)
            rtn += '\n {}: {}\n'.format(ged_sim_str, ged_str)
        else:
            rtn += '\n {}: {:.2f}\n'.format(ged_sim_str, ged_sim)
        t = r.time(qid, gid)
        if t:
            rtn += 'time: {:.2f} sec'.format(t)
        else:
            rtn += 'time: -'
    return rtn


def get_ged_select_norm_str(r, qid, gid, norm):
    ged = r.dist_sim(qid, gid, norm=False)[1]
    norm_ged = r.dist_sim(qid, gid, norm=True)[1]
    if norm:
        return '{:.2f} ({})'.format(norm_ged, ged)
    else:
        return '{} ({:.2f})'.format(ged, norm_ged)


def get_graph_stats_text(g):
    return '\n#nodes: {}\n#edges: {}\ndensity: {:.2f}'.format( \
        g.number_of_nodes(), g.number_of_edges(), nx.density(g))


exp7()
