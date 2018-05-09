from vis import *

info_dict = {
    'draw_node_size': 10,
    'draw_node_label_enable': True,
    'draw_node_label_font_size': 8,

    'draw_edge_label_enable': False,
    'draw_edge_label_font_size': 6
}

q = nx.read_gexf('AIDS/0.gexf')

gs = []
for i in range(1, 10):
    print(i)
    gs.append(nx.read_gexf('AIDS/' + str(i) + '.gexf'))
    vis(q, gs, info_dict)

plt.savefig('plot', dpi=200)
    
