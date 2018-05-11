from vis import *
from utils import get_data

info_dict = {
    'draw_node_size': 10,
    'draw_node_label_enable': True,
    'draw_node_label_font_size': 8,
    'draw_node_color_map': {'C': 'red',
           'O': 'blue',
           'N': 'green'},
   
    'draw_edge_label_enable': False,
    'draw_edge_label_font_size': 6,

    'each_graph_text_list': ["test1", "test2", "test3", "test4", "test5", "test6", "test7", "test8", "test9", "test10"],
    'each_graph_font_size': 10,

    'plot_dpi': 200,
    'plot_save_path': 'plot2'
}


test_data = get_data('aids10k', train=False)
train_data = get_data('aids10k', train=True)
q = test_data.graphs[0]

gs = []
for i in range(1, 5):
    print(i)
    gs.append(train_data.graphs[i])

vis(q, gs, info_dict)

'''
TODO:
1. node color [done]
2. support graph-level text [done]
3. plt.save in vis [done]
4. edge color
'''
