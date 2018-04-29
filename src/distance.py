from utils import get_root_path
from nx_to_gxl import nx_to_gxl
from ged4py.algorithm import graph_edit_dist
from os import system
import networkx as nx


def hungarian_ged(g1, g2):
    # https://github.com/Jacobe2169/ged4py
    return graph_edit_dist.compare(g1, g2)


def astar_ged(g1, g2):
    # https://github.com/dan-zam/graph-matching-toolkit
    gp = get_gmt_path()
    src, tp = setup_temp_folder(gp)
    meta1 = write_to_temp(g1, tp, 'g1')
    meta2 = write_to_temp(g2, tp, 'g2')
    if meta1 != meta2:
        raise RuntimeError('Different meta data {} vs {}'.format(meta1, meta2))
    setup_property_file(src, gp, meta1)
    exec(
        'cd {} && java -classpath graph-matching-toolkit.jar algorithms.GraphMatching properties/properties_temp.prop'.format(
            gp))
    return get_result(gp)


def setup_temp_folder(gp):
    tp = gp + '/data/temp'
    exec('rm -rf {} && mkdir {}'.format(tp, tp))
    src = get_root_path() + '/src/gmk_files'
    exec('cp {}/temp.xml {}/temp.xml'.format(src, tp))
    return src, tp


def setup_property_file(src, gp, meta):
    exec('cp {}/{}.prop {}/properties/properties_temp.prop'. \
         format(src, meta, gp))


def write_to_temp(g, tp, name):
    node_attres, edge_attrs = nx_to_gxl(g, name, '{}/{}.gxl'.format(tp, name))
    return 'astar_' + '_'.join(list(node_attres.keys()) + list( \
        edge_attrs.keys()))


def get_result(gp):
    with open('{}/result/temp'.format(gp)) as f:
        lines = f.readlines()
        return (float(lines[22]) + float(lines[23])) # alpha=0.5 --> already /2


def get_gmt_path():
    return get_root_path() + '/src/graph-matching-toolkit'


def exec(cmd):
    print(cmd)
    system(cmd)
