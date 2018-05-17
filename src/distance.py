from utils import get_root_path, exec, get_ts
from nx_to_gxl import nx_to_gxl
from os.path import isfile
import fileinput


def mcs(g1, g2):
    # TODO: implementation by Hao Ding
    return 0


def hungarian_ged(g1, g2):
    # # https://github.com/Jacobe2169/ged4py
    # return graph_edit_dist.compare(g1, g2)
    return ged(g1, g2, 'hungarian')


def astar_ged(g1, g2):
    return ged(g1, g2, 'astar')


def beam_ged(g1, g2, s):
    return ged(g1, g2, 'beam{}'.format(s))


def vj_ged(g1, g2):
    return ged(g1, g2, 'vj')


def ged(g1, g2, algo):
    # https://github.com/dan-zam/graph-matching-toolkit
    gp = get_gmt_path()
    src, tp = setup_temp_folder(gp)
    meta1 = write_to_temp(g1, tp, algo, 'g1')
    meta2 = write_to_temp(g2, tp, algo, 'g2')
    if meta1 != meta2:
        raise RuntimeError(
            'Different meta data {} vs {}'.format(meta1, meta2))
    setup_property_file(src, gp, meta1)
    if not exec(
            'cd {} && java -classpath {}/src/graph-matching-toolkit/bin algorithms.GraphMatching ./properties/properties_temp_{}.prop'.format(
                gp, get_root_path(), get_ts())):
        return -1
    return get_result(gp, algo)


def setup_temp_folder(gp):
    tp = gp + '/data/temp_' + get_ts()
    exec('rm -rf {} && mkdir {}'.format(tp, tp))
    src = get_root_path() + '/src/gmt_files'
    exec('cp {}/temp.xml {}/temp_{}.xml'.format(src, tp, get_ts()))
    return src, tp


def setup_property_file(src, gp, meta):
    destfile = '{}/properties/properties_temp_{}.prop'.format(gp, get_ts())
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
            print(line.replace('temp', 'temp_' + get_ts()))


def write_to_temp(g, tp, algo, g_name):
    node_attres, edge_attrs = nx_to_gxl(g, g_name,
                                        '{}/{}.gxl'.format(tp, g_name))
    return algo + '_' + '_'.join(list(node_attres.keys()) + list( \
        edge_attrs.keys()))


def get_result(gp, algo):
    with open('{}/result/temp_{}'.format(gp, get_ts())) as f:
        lines = f.readlines()
        ln = 23 if 'beam' in algo else 22
        rtn = float(lines[ln]) * 2  # alpha=0.5 --> / 2
        assert (rtn - int(rtn) == 0)
        rtn = int(rtn)
        if rtn < 0:
            rtn = -1  # in case rtn == -2
        return rtn


def get_gmt_path():
    return get_root_path() + '/src/graph-matching-toolkit'


if __name__ == '__main__':
    from utils import load_data

    test_data = load_data('aids10k', train=False)
    train_data = load_data('aids10k', train=True)
    g1 = train_data.graphs[0]
    g2 = test_data.graphs[0]
    print(mcs(g1, g2))

    g1 = test_data.graphs[15]
    g2 = train_data.graphs[761]
    import networkx as nx

    # nx.write_gexf(g1, get_root_path() + '/temp/g1.gexf')
    # nx.write_gexf(g2, get_root_path() + '/temp/g2.gexf')
    g1 = nx.read_gexf(get_root_path() + '/temp/g1_small.gexf')
    g2 = nx.read_gexf(get_root_path() + '/temp/g2_small.gexf')
    print(astar_ged(g1, g2))
    print(beam_ged(g1, g2, 2))
