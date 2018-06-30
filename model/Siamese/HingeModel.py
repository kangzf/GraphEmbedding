from config import FLAGS
from models import Model


class SiameseGCNTNHinge(Model):
    def __init__(self, input_dim):
        assert (FLAGS.loss_func == 'hinge')
        self.input_dim = input_dim
        # TODO
        # Build the model.
        super(SiameseGCNTNHinge, self).__init__()

    def apply_final_act_np(self, score):
        # TODO
        return None

    def get_feed_dict(self, data, dist_calculator, tvt, test_id, train_id):
        rtn = dict()
        # TODO
        return rtn

    def _get_ins(self, layer, tvt):
        assert (layer.__class__.__name__ == 'GraphConvolution')
        ins = []
        # TODO
        return ins

    def _proc_ins(self, ins, k, layer, tvt):
        return None  # TODO

    def _proc_outs(self, outs, k, layer, tvt):
        return None  # TODO

    def _hinge_loss(self):
        #     y_pred = self.pred_sim()
        #     pos_interact_score = y_pred[
        #                          :FLAGS.batch_size_p]  # need set new flag for positive sample number & assume pred is a vector
        #     neg_interact_score = y_pred[FLAGS.batch_size_p:]
        #     diff_mat = tf.reshape(tf.tile(pos_interact_score, [FLAGS.num_negatives]),
        #                           # need set new flag for negative sampling
        #                           (-1,
        #                            1)) - neg_interact_score  # assume negative sampling is conducted in this way: p+n1+n2+..+nk
        #     rank_loss = tf.reduce_mean(-tf.log(tf.sigmoid(gamma * diff_mat)))
        #     return rank_loss
        return None  # TODO
