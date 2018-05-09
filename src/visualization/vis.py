import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import math

# matplotlib.rcParams.update({'figure.autolayout': True})

def info_dict_preprocess(info_dict):
    info_dict.setdefault('draw_node_size', 10)
    info_dict.setdefault('draw_node_label_enable', True)
    info_dict.setdefault('draw_node_label_font_size', 6)
    info_dict.setdefault('draw_edge_label_enable', False)
    info_dict.setdefault('draw_edge_label_font_size', 6)


def calc_subplot_size(area):
    h = int(math.sqrt(area))
    while area % h != 0:
        area += 1
    w = area / h
    return [h, w]


def draw_extra():
    pass
    plt.axis('off')


def draw_graph(g, info_dict):
    if g is None:
        return
    pos = graphviz_layout(g)
    
    nx.draw_networkx(g, pos, node_color='y', with_labels=False, node_size=info_dict['draw_node_size'])

    if info_dict['draw_node_label_enable'] == True:
        node_labels = nx.get_node_attributes(g, 'type')
        nx.draw_networkx_labels(g, pos, node_labels, font_size=info_dict['draw_node_label_font_size'])
    
    if info_dict['draw_edge_label_enable'] == True:
        edge_labels = nx.get_edge_attributes(g, 'valence')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=info_dict['draw_edge_label_font_size'])

    draw_extra()


def vis(q=None, gs=None, info_dict=None):

    info_dict_preprocess(info_dict)

    # get num
    graph_num = 1 + len(gs)
    plot_m, plot_n = calc_subplot_size(graph_num)

    # draw query graph
    plt.subplot(plot_m, plot_n, 1)
    draw_graph(q, info_dict)

    # draw graph candidates
    for i in range(len(gs)):
        plt.subplot(plot_m, plot_n, i + 2)
        draw_graph(gs[i], info_dict)

    # plot setting
    # plt.tight_layout()
    left  = 0.01  # the left side of the subplots of the figure
    right = 0.99    # the right side of the subplots of the figure
    bottom = 0.01   # the bottom of the subplots of the figure
    top = 0.99      # the top of the subplots of the figure
    wspace = 0   # the amount of width reserved for blank space between subplots
    hspace = 0   # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

