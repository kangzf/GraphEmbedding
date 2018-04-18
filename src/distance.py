from ged4py.algorithm import graph_edit_dist

def GED(g1, g2):
    # https://github.com/Jacobe2169/ged4py
    return graph_edit_dist.compare(g1, g2)