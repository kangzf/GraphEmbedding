import sys
sys.path.append('..')
from utils import get_root_path
import networkx as nx


dirin = get_root_path() + '/data'
file = dirin + '/AIDO99SD'
dirout = dirin + '/AIDS'


g = nx.Graph()
line_i = 0
gid = 0
with open(file) as f:
    for line in f:
        if '$$$$' in line:
            print(gid)
            nx.write_gexf(g, dirout + '/{}.gexf'.format(gid))
            g = nx.Graph()
            line_i = 0
            gid += 1
        else:
            ls = line.rstrip().split()
            if len(ls) == 9:
                nid = line_i - 4
                type = line.rstrip().split()[3]
                if type != 'H':
                    g.add_node(nid, type=type)
            elif len(ls) == 6:
                ls = line.rstrip().split()
                nid0 = int(ls[0]) - 1
                nid1 = int(ls[1]) - 1
                valence = int(ls[2])
                if nid0 in g.nodes() and nid1 in g.nodes():
                    g.add_edge(nid0, nid1, valence=valence)
            line_i += 1



