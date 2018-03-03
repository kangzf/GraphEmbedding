import os
import networkx as nx
import numpy as np
import re

numbers = re.compile(r'(\d+)')
def num_sort(value):
    tokens = numbers.split(value)
    tokens[1::2] = map(int, tokens[1::2])
    return tokens

def get_files(dirname, extn, max_files=0):
    all_files = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(extn)]
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if f.endswith(extn):
                all_files.append(os.path.join(root, f))

    all_files = list(set(all_files))
    all_files.sort(key=num_sort)
    # print('all_files:', all_files)

    if max_files:
        return all_files[:max_files]
    else:
        return all_files


# dirname -> list of networkx graph
def read_graphs(dirname):
    fnames = get_files(dirname, 'gexf')
    graphs = [nx.read_gexf(fname) for fname in fnames]
    return graphs
