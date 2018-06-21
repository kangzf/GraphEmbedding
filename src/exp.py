#!/usr/bin/env python3
from utils import get_result_path, create_dir_if_not_exists, load_data, \
    get_ts, exec_turnoff_print, prompt, prompt_get_computer_name, \
    check_nx_version, prompt_get_cpu, format_float
from metrics import Metric, precision_at_ks, mean_reciprocal_rank, \
    mean_squared_error, average_time
from distance import ged
from similarity import create_sim_kernel
from results import load_results_as_dict, load_result
import networkx as nx

check_nx_version()
import multiprocessing as mp
from time import time
from random import randint, uniform

from pandas import read_csv
import matplotlib.pyplot as plt
import matplotlib
from vis import vis
import numpy as np

BASELINE_MODELS = ['beam5', 'beam10', 'beam20', 'beam40', 'beam80', \
                   'hungarian', 'vj']
TRUE_MODEL = 'beam80'

""" Plotting args. """
args1 = {'astar': {'color': 'grey'},
         'beam5': {'color': 'deeppink'},
         'beam10': {'color': 'b'},
         'beam20': {'color': 'forestgreen'},
         'beam40': {'color': 'darkorange'},
         'beam80': {'color': 'cyan'},
         'hungarian': {'color': 'deepskyblue'},
         'vj': {'color': 'darkcyan'},
         'graph2vec': {'color': 'darkcyan'},
         'siamese_gcntn': {'color': 'red'}}
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
         'vj': {'marker': 'h', 'facecolors': 'none',
                'edgecolors': 'darkcyan'},
         'graph2vec': {'marker': 'h', 'facecolors': 'none',
                       'edgecolors': 'darkcyan'},
         'siamese_gcntn': {'marker': 'P', \
                           'facecolors': 'none', 'edgecolors': 'red'}
         }
font = {'family': 'serif',
        'size': 22}
matplotlib.rc('font', **font)


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
    dataset = prompt('Which dataset?')
    model = prompt('Which model?')
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
    computer_name = prompt_get_computer_name()
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
    models = BASELINE_MODELS
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
    dataset = 'aids50'
    models = BASELINE_MODELS
    metric = 'ap@k'
    norms = [True, False]
    rs = load_results_as_dict(dataset, models)
    true_result = rs[TRUE_MODEL]
    plot_apk(dataset, models, rs, true_result, metric, norms)


def plot_apk(dataset, models, rs, true_result, metric, norms, plot_results=True):
    """ Plot ap@k. """
    create_dir_if_not_exists('{}/{}/{}'.format( \
        get_result_path(), dataset, metric))
    rtn = {}
    for norm in norms:
        ks = []
        k = 1
        _, n = true_result.m_n()
        while k < n:
            ks.append(k)
            k *= 2
        plot_apk_helper(
            dataset, models, rs, true_result, metric, norm, ks,
            True, plot_results)
        ks = range(1, n)
        d = plot_apk_helper(
            dataset, models, rs, true_result, metric, norm, ks,
            False, plot_results)
        rtn.update(d)
    return rtn


def plot_apk_helper(dataset, models, rs, true_result, metric, norm, ks,
                    logscale, plot_results):
    print_ids = []
    rtn = {}
    for model in models:
        aps = precision_at_ks(true_result, rs[model], norm, ks, print_ids)
        rtn[model] = {'ks': ks, 'aps': aps}
        # print('aps {}: {}'.format(model, aps))
    rtn = {'apk{}'.format(get_norm_str(norm)): rtn}
    if not plot_results:
        return rtn
    plt.figure(figsize=(16, 10))
    for model in models:
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
    sp = get_result_path() + '/{}/{}/ged_{}_{}_{}_{}{}.png'.format( \
        dataset, metric, metric, dataset, '_'.join(models), kss,
        get_norm_str(norm))
    plt.savefig(sp)
    print('Saved to {}'.format(sp))
    return rtn


def get_norm_str(norm):
    if norm is None:
        return ''
    elif norm:
        return '_norm'
    else:
        return '_nonorm'


def exp7():
    dataset = 'aids50'
    models = BASELINE_MODELS
    metric = 'time'
    sim_kernel = 'gaussian'
    yeta = 1.0
    norms = [True, False]
    rs = load_results_as_dict(dataset, models)
    true_result = rs[TRUE_MODEL]
    plot_mrr_mse_time(
        dataset, models, rs, true_result, metric, norms, sim_kernel, yeta)


def plot_mrr_mse_time(dataset, models, rs, true_result, metric, norms,
                      sim_kernel, yeta, plot_results=True):
    """ Plot mrr or mse. """
    create_dir_if_not_exists('{}/{}/{}'.format(
        get_result_path(), dataset, metric))
    rtn = {}
    for norm in norms:
        d = plot_mrr_mse_time_helper(
            dataset, models, rs, true_result, metric, norm, sim_kernel, yeta,
            plot_results)
        rtn.update(d)
    return rtn


def plot_mrr_mse_time_helper(dataset, models, rs, true_result, metric, norm,
                             sim_kernel, yeta, plot_results):
    print_ids = []
    rtn = {}
    mrr_mse_list = []
    for model in models:
        if metric == 'mrr':
            mrr_mse_time = mean_reciprocal_rank(
                true_result, rs[model], norm, print_ids)
        elif metric == 'mse':
            mrr_mse_time = mean_squared_error(
                true_result, rs[model], sim_kernel, yeta, norm)
        elif metric == 'time':
            mrr_mse_time = average_time(rs[model])
        else:
            raise RuntimeError('Unknown {}'.format(metric))
        # print('{} {}: {}'.format(metric, model, mrr_mse_time))
        rtn[model] = mrr_mse_time
        mrr_mse_list.append(mrr_mse_time)
    rtn = {'{}{}'.format(metric, get_norm_str(norm)): rtn}
    if not plot_results:
        return rtn
    plt.figure(figsize=(16, 10))
    ind = np.arange(len(mrr_mse_list))  # the x locations for the groups
    width = 0.35  # the width of the bars
    bars = plt.bar(ind, mrr_mse_list, width)
    for i, bar in enumerate(bars):
        bar.set_color(args1[models[i]]['color'])
    autolabel(bars)
    plt.xlabel('model')
    plt.xticks(ind, models)
    if metric == 'time':
        ylabel = 'time (msec)'
        norm = None
    else:
        ylabel = metric
    plt.ylabel(ylabel)
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    sp = get_result_path() + '/{}/{}/ged_{}_{}_{}{}.png'.format( \
        dataset, metric, metric, dataset, '_'.join(models),
        get_norm_str(norm))
    plt.savefig(sp)
    print('Saved to {}'.format(sp))
    return rtn


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text( \
            rect.get_x() + rect.get_width() / 2., 1.005 * height, \
            format_float(height), ha='center', va='bottom')


def exp8():
    """ Query visualization. """
    dataset = 'aids10k'
    model = 'graph2vec'
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
    tr = load_result(dataset, TRUE_MODEL)
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
                '/{}/query_vis/{}/query_vis_{}_{}_{}{}.png'.format( \
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


def exp9():
    """ Check similarity kernel. """
    dataset = 'aids50'
    model = 'beam80'
    sim_kernel_name = 'gaussian'
    norms = [True, False]
    yetas1 = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    yetas2 = np.arange(0.1, 1.1, 0.1)
    print(yetas2)
    yetas3 = get_gaussian_yetas(0.0001, 0.001)
    result = load_result(dataset, model)
    for norm in norms:
        for yetas in [yetas1, sorted(yetas2), sorted(yetas3)]:
            sim_kernels = []
            for yeta in yetas:
                sim_kernels.append(create_sim_kernel(sim_kernel_name, yeta))
            plot_sim_kernel(dataset, result, sim_kernels, norm)


def get_gaussian_yetas(middle_yeta, delta_yeta):
    yetas2 = [middle_yeta]
    for i in range(1, 6):
        yetas2.append(middle_yeta + i * delta_yeta)
        yetas2.append(middle_yeta - i * delta_yeta)
    return yetas2


def plot_sim_kernel(dataset, result, sim_kernels, norm):
    dir = '{}/{}/sim'.format(get_result_path(), dataset)
    create_dir_if_not_exists(dir)
    m, n = result.m_n()
    ged_mat = result.dist_sim_mat(norm=norm)
    plt.figure(figsize=(16, 10))
    for sim_kernel in sim_kernels:
        for i in range(m):
            for j in range(n):
                d = ged_mat[i][j]
                plt.scatter(ged_mat[i][j], sim_kernel.dist_to_sim(d), s=100)
        # Plot the function.
        xs, ys = get_sim_kernel_points(ged_mat, sim_kernel)
        plt.plot(xs, ys, label=sim_kernel.name())
    plt.xlabel('GED')
    plt.ylabel('Similarity')
    plt.ylim([-0.06, 1.06])
    plt.legend(loc='best')
    plt.grid(linestyle='dashed')
    plt.tight_layout()
    # plt.show()
    sp = '{}/sim_{}_{}{}.png'.format(
        dir, dataset, '_'.join(
            [sk.shortname() for sk in sim_kernels]), get_norm_str(norm))
    plt.savefig(sp)
    print('Saved to {}'.format(sp))


def get_sim_kernel_points(ged_mat, sim_kernel):
    xs = []
    i = 0
    while i < np.amax(ged_mat) * 1.05:
        xs.append(i)
        i += 0.1
    ys = [sim_kernel.dist_to_sim(x) for x in xs]
    return xs, ys


if __name__ == '__main__':
    exp6()
