from data import Data
from config import FLAGS
from utils import load_data, exec_turnoff_print
from graphs import ModelGraphList, NodeFeatureOneHotEncoder

exec_turnoff_print()


class SiameseModelData(Data):
    def __init__(self):
        self.dataset = FLAGS.dataset
        self.valid_percentage = FLAGS.valid_percentage
        self.node_feat_name = FLAGS.node_feat_name
        self.node_feat_encoder = FLAGS.node_feat_encoder
        self.edge_feat_name = FLAGS.edge_feat_name
        self.edge_feat_processor = FLAGS.edge_feat_processor
        self.sampler = FLAGS.sampler
        self.sample_num = FLAGS.sample_num
        self.sampler_duplicate_removal = FLAGS.sampler_duplicate_removal
        super().__init__(self._get_name())
        print('{} train graphs; {} validation graphs; {} test graphs'.format( \
            self.train_data.num_graphs(),
            self.valid_data.num_graphs(),
            self.test_data.num_graphs()))

    def init(self):
        orig_train_data = load_data(self.dataset, train=True)
        self.n = len(orig_train_data.graphs)
        train_gs, valid_gs = self._train_val_split(orig_train_data)
        test_gs = load_data(self.dataset, train=False).graphs
        self.node_feat_encoder = self._get_node_feature_encoder(
            orig_train_data.graphs + test_gs)
        self._check_graphs_num(test_gs, 'test')
        self.train_data = ModelGraphList(
            self.sampler, self.sample_num, self.sampler_duplicate_removal,
            train_gs, self.node_feat_encoder)
        self.valid_data = ModelGraphList(
            self.sampler, self.sample_num, self.sampler_duplicate_removal,
            valid_gs, self.node_feat_encoder)
        self.test_data = ModelGraphList(
            self.sampler, self.sample_num, self.sampler_duplicate_removal,
            test_gs, self.node_feat_encoder)
        self.m = self.test_data.num_graphs()

        assert (len(train_gs) + len(valid_gs) == len(orig_train_data.graphs))

    def input_dim(self):
        return self.node_feat_encoder.input_dim()

    def get_graph_pair(self, train_val_test):
        graph_collection = self._get_graph_collection(train_val_test)
        return graph_collection.get_graph_pair()

    def get_orig_train_graph(self, orig_train_id):
        trainlen = self.train_data.num_graphs()
        vallen = self.valid_data.num_graphs()
        if 0 <= orig_train_id < trainlen:
            return self.train_data.get_graph(orig_train_id)
        elif orig_train_id < trainlen + vallen:
            return self.valid_data.get_graph(orig_train_id - trainlen)
        else:
            assert (False)

    def get_dist(self, g1, g2, dist_calculator):
        return dist_calculator.calculate_dist(g1, g2)

    def m_n(self):
        return self.m, self.n

    def _get_name(self):
        li = []
        for k, v in sorted(self.__dict__.items(), key=lambda x: x[0]):
            li.append('{}'.format(v))
        return '_'.join(li)

    def _train_val_split(self, orig_train_data):
        if self.valid_percentage < 0 or self.valid_percentage > 1:
            raise RuntimeError('valid_percentage {} must be in [0, 1]'.format(
                self.valid_percentage))
        gs = orig_train_data.graphs
        sp = int(len(gs) * (1 - self.valid_percentage))
        train_graphs = gs[0:sp]
        valid_graphs = gs[sp:]
        self._check_graphs_num(train_graphs, 'train')
        self._check_graphs_num(valid_graphs, 'validation')
        return train_graphs, valid_graphs

    def _check_graphs_num(self, graphs, label):
        if len(graphs) <= 2:
            raise RuntimeError('Insufficient {} graphs {}'.format( \
                label, len(graphs)))

    def _get_node_feature_encoder(self, gs):
        if self.node_feat_encoder == 'onehot':
            return NodeFeatureOneHotEncoder(gs, self.node_feat_name)
        else:
            raise RuntimeError('Unknown node_feat_encoder {}'.format(
                self.node_feat_encoder))

    def _get_graph_collection(self, train_val_test):
        if train_val_test == 'train':
            return self.train_data
        elif train_val_test == 'val':
            return self.valid_data
        elif train_val_test == 'test':
            return self.test_data
        else:
            raise RuntimeError('Unknown train_val_test {}'.format(
                train_val_test))
