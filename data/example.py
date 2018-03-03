import dataset_util as util

dirname = './enzymes/graph'
# dirname = './collab'
graphs = util.read_graphs(dirname)
print(graphs[0].number_of_nodes())
print(graphs[0].number_of_edges())
