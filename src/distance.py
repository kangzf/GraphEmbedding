from utils import get_root_path, exec, get_ts
from nx_to_gxl import nx_to_gxl
from os.path import isfile
from os import getpid
from time import time
import fileinput
import networkx as nx

def mcs(g1, g2):
    nx.write_gexf(g1, 'temp_1.gexf')
    nx.write_gexf(g2, 'temp_2.gexf')
    # Force using bash instead of dash.
    # `source activate` does not work on dash.
    # graphembedding is a virtual python environment with networkx==2.0.
    # By default networkx==1.10 is assumed.
    cmd = 'source activate graphembedding && python mcs_cal.py'
    exec('/bin/bash -c "{}"'.format(cmd))
    f = open('mcs_result.txt','r')
    return int(f.read())
    return 0


def ged(g1, g2, algo, debug=False, timeit=False):
    # https://github.com/dan-zam/graph-matching-toolkit
    gp = get_gmt_path()
    src, tp = setup_temp_folder(gp)
    meta1 = write_to_temp(g1, tp, algo, 'g1')
    meta2 = write_to_temp(g2, tp, algo, 'g2')
    if meta1 != meta2:
        raise RuntimeError(
            'Different meta data {} vs {}'.format(meta1, meta2))
    setup_property_file(src, gp, meta1)
    if timeit:
        t = time()
    rtn = []
    if not exec(
            'cd {} && java -classpath {}/src/graph-matching-toolkit/bin algorithms.GraphMatching ./properties/properties_temp_{}_{}.prop'.format(
                gp, get_root_path(), get_ts(), getpid(), timeout=1000)):
        rtn.append(-1)
    else:
        d, lcnt, g1size, g2size = get_result(gp, algo)
        rtn.append(d)
        if g1size != g1.number_of_nodes():
            print('g1size {} g1.number_of_nodes() {}'.format(g1size, g1.number_of_nodes()))
        assert (g1size == g1.number_of_nodes())
        assert (g2size == g2.number_of_nodes())
    if debug:
        rtn += [lcnt, g1, g2]
    if timeit:
        rtn.append(time() - t)
    return tuple(rtn)


def setup_temp_folder(gp):
    tp = gp + '/data/temp_{}_{}'.format(get_ts(), getpid())
    exec('rm -rf {} && mkdir {}'.format(tp, tp))
    src = get_root_path() + '/src/gmt_files'
    exec('cp {}/temp.xml {}/temp_{}_{}.xml'.format(src, tp, get_ts(), getpid()))
    return src, tp


def setup_property_file(src, gp, meta):
    destfile = '{}/properties/properties_temp_{}_{}.prop'.format( \
        gp, get_ts(), getpid())
    srcfile = '{}/{}.prop'.format(src, meta)
    if not isfile(srcfile):
        if 'beam' in meta: # for beam
            metasp = meta.split('_')
            s = int(metasp[0][4:])
            if s <= 0:
                raise RuntimeError('Invalid s for beam search: {}'.format(s))
            newmeta = '_'.join(['beam'] + metasp[1:])
            srcfile = '{}/{}.prop'.format(src, newmeta)
        else:
            raise RuntimeError('File {} does not exist'.format(srcfile))
    exec('cp {} {}'.format(srcfile, destfile))
    for line in fileinput.input(destfile, inplace=True):
        line = line.rstrip()
        if line == 's=': # for beam
            print('s={}'.format(s))
        else:
            print(line.replace('temp', 'temp_{}_{}'.format(get_ts(), getpid())))


def write_to_temp(g, tp, algo, g_name):
    node_attres, edge_attrs = nx_to_gxl(g, g_name,
                                        '{}/{}.gxl'.format(tp, g_name))
    return algo + '_' + '_'.join(sorted(list(node_attres.keys())) + \
                                 sorted(list(edge_attrs.keys())))


def get_result(gp, algo):
    with open('{}/result/temp_{}_{}'.format(gp, get_ts(), getpid())) as f:
        lines = f.readlines()
        ln = 23 if 'beam' in algo else 22
        d = float(lines[ln]) * 2  # alpha=0.5 --> / 2
        assert (d - int(d) == 0)
        d = int(d)
        if d < 0:
            d = -1  # in case rtn == -2
        ln = 26 if 'beam' in algo else 25
        g1size = int(lines[ln])
        ln = 27 if 'beam' in algo else 26
        g2size = int(lines[ln])
        ln = 28 if 'beam' in algo else 27
        lcnt = float(lines[ln])
        return d, lcnt, g1size, g2size


def get_gmt_path():
    return get_root_path() + '/src/graph-matching-toolkit'


if __name__ == '__main__':
    from utils import load_data

    test_data = load_data('aids10k', train=False)
    train_data = load_data('aids10k', train=True)
    g1 = train_data.graphs[0]
    g2 = test_data.graphs[0]
    print(mcs(g1, g2))

    # g1 = test_data.graphs[15]
    # g2 = train_data.graphs[761]
    #
    # # nx.write_gexf(g1, get_root_path() + '/temp/g1.gexf')
    # # nx.write_gexf(g2, get_root_path() + '/temp/g2.gexf')
    # g1 = nx.read_gexf(get_root_path() + '/temp/g1_small.gexf')
    # g2 = nx.read_gexf(get_root_path() + '/temp/g2_small.gexf')
    # print(astar_ged(g1, g2))
    # print(beam_ged(g1, g2, 2))
