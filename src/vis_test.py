from vis import *
from utils import get_data

info_dict = {
    'draw_node_size': 10,
    'draw_node_label_enable': True,
    'draw_node_label_font_size': 8,

    'draw_edge_label_enable': True,
    'draw_edge_label_font_size': 6,
    'sim_score': [],
    'sim_score_enable': True
}


test_data = get_data('aids10k', train=False)
train_data = get_data('aids10k', train=True)
q = test_data.graphs[0]

gs = []
for i in range(1, 5):
    print(i)
    gs.append(train_data.graphs[i])

vis(q, gs, info_dict)
plt.savefig('plot', dpi=200)

'''
TODO:
1. node color
2. support graph-level text
3. plt.save in vis
4. edge color
'''