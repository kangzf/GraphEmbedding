def get_model_fun(model, train):
    import sys
    sys.path.insert(0, get_root_path())
    if model == 'iwge':
        if train:
            from IWGE.src.train import train
            return train
        else:
            from IWGE.src.test import test
            return test
    else:
        raise RuntimeError('Not recognized model %s' % model)


def get_data(data, train):
    if data == 'syn':
        from data import SynData
        return SynData(train)
    elif data == 'protein':
        from data import ProteinData
        return ProteinData(train)
    else:
        raise RuntimeError('Not recognized data %s' % data)


def get_root_path():
    from os.path import dirname, abspath
    return dirname(dirname(abspath(__file__)))

def draw_graph(g, file):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure()
    import networkx as nx
    nx.draw(g, ax=f.add_subplot(111))
    f.savefig(file)
    print('Saved graph to {}'.format(file))