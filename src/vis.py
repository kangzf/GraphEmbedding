import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import math


# matplotlib.rcParams.update({'figure.autolayout': True})

def info_dict_preprocess(info_dict):
    info_dict.setdefault('draw_node_size', 10)
    info_dict.setdefault('draw_node_label_enable', True)
    info_dict.setdefault('node_label_name', '')
    info_dict.setdefault('draw_node_label_font_size', 6)

    info_dict.setdefault('draw_edge_label_enable', False)
    info_dict.setdefault('edge_label_name', '')
    info_dict.setdefault('draw_edge_label_font_size', 6)

    info_dict.setdefault('each_graph_text_font_size', "")
    info_dict.setdefault('each_graph_text_pos', [0.5, 0.8])

    info_dict.setdefault('plot_dpi', 200)
    info_dict.setdefault('plot_save_path', "")

    info_dict.setdefault('top_space', 0.08)
    info_dict.setdefault('bottom_space', 0)
    info_dict.setdefault('hbetween_space', 0.5)
    info_dict.setdefault('wbetween_space', 0.01)


def calc_subplot_size(area):
    h = int(math.sqrt(area))
    while area % h != 0:
        area += 1
    w = area / h
    return [h, w]


def list_safe_get(l, index, default):
    try:
        return l[index]
    except IndexError:
        return default


def draw_extra(i, ax, info_dict, text):
    pass
    left = list_safe_get(info_dict['each_graph_text_pos'], 0, 0.5)
    bottom = list_safe_get(info_dict['each_graph_text_pos'], 1, 0.8)
    # print(left, bottom)
    ax.title.set_position([left, bottom])
    ax.set_title(text, fontsize=info_dict['each_graph_text_font_size'])
    plt.axis('off')


def draw_graph(g, info_dict):
    if g is None:
        return
    pos = graphviz_layout(g)

    node_labels = nx.get_node_attributes(g, 'type')
    color_values = [info_dict['draw_node_color_map'].get(node_label, 'yellow')
                    for node_label in node_labels.values()]
    nx.draw_networkx(g, pos, node_color=color_values, with_labels=False,
                     node_size=info_dict['draw_node_size'])

    if info_dict['draw_node_label_enable'] == True:
        node_labels = nx.get_node_attributes(g, info_dict['node_label_name'])

        nx.draw_networkx_labels(g, pos, node_labels, font_size=info_dict[
            'draw_node_label_font_size'])

    if info_dict['draw_edge_label_enable'] == True:
        edge_labels = nx.get_edge_attributes(g, info_dict['edge_label_name'])
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                     font_size=info_dict[
                                         'draw_edge_label_font_size'])


def vis(q=None, gs=None, info_dict=None):
    plt.figure()
    info_dict_preprocess(info_dict)

    # get num
    graph_num = 1 + len(gs)
    plot_m, plot_n = calc_subplot_size(graph_num)

    # draw query graph
    ax = plt.subplot(plot_m, plot_n, 1)
    draw_graph(q, info_dict)
    draw_extra(0, ax, info_dict,
               list_safe_get(info_dict['each_graph_text_list'], 0, ""))

    # draw graph candidates
    for i in range(len(gs)):
        ax = plt.subplot(plot_m, plot_n, i + 2)
        draw_graph(gs[i], info_dict)
        draw_extra(i, ax, info_dict,
                   list_safe_get(info_dict['each_graph_text_list'], i + 1, ""))

    # plot setting
    # plt.tight_layout()
    left = 0.01  # the left side of the subplots of the figure
    right = 0.99  # the right side of the subplots of the figure
    top = 1 - info_dict['top_space']  # the top of the subplots of the figure
    bottom = \
        info_dict['bottom_space']  # the bottom of the subplots of the figure
    wspace = \
        info_dict['wbetween_space']  # the amount of width reserved for blank space between subplots
    hspace = \
        info_dict['hbetween_space']  # the amount of height reserved for white space between subplots

    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                        wspace=wspace, hspace=hspace)

    # save / display
    save_path = info_dict['plot_save_path']
    if save_path is None or save_path == "":
        plt.show()
    else:
        sp = info_dict['plot_save_path']
        print('Saving qeury vis plot to {}'.format(sp))
        plt.savefig(sp, dpi=info_dict['plot_dpi'])
