from utils import get_root_path, exec, get_ts
from nx_to_gxl import nx_to_gxl
import fileinput


def mcs(g1, g2):
    nx.write_gexf(g1, 'temp_1.gexf')
    nx.write_gexf(g2, 'temp_2.gexf')

    exec('source activate graphembedding && python mcs_cal.py')

    f = open('mcs_result.txt','r')
    return int(f.read())
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
    src = get_root_path() + '/src/gmk_files'
    exec('cp {}/temp.xml {}/temp_{}.xml'.format(src, tp, get_ts()))
    return src, tp


def setup_property_file(src, gp, meta):
    file = '{}/properties/properties_temp_{}.prop'.format(gp, get_ts())
    exec('cp {}/{}.prop {}'.format(src, meta, file))
    for line in fileinput.input(file, inplace=True):
        print(line.rstrip().replace('temp', 'temp_' + get_ts()))


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
