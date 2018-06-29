from config import FLAGS
from utils import get_ts, create_dir_if_not_exists
from utils_siamese import get_siamese_dir, get_model_info_as_str
import tensorflow as tf


class Saver(object):
    def __init__(self, sess):
        if FLAGS.log:
            model_str = self._get_model_str()
            logdir = '{}/logs/{}_{}'.format(
                get_siamese_dir(), model_str, get_ts())
            create_dir_if_not_exists(logdir)
            self.tw = tf.summary.FileWriter(logdir + '/train', sess.graph)
            self.vw = tf.summary.FileWriter(logdir + '/val', sess.graph)
            self.merged = tf.summary.merge_all()
            self._log_model_info(logdir, sess)
            print('Logging to {}'.format(logdir))

    def proc_objs(self, objs, tvt):
        if FLAGS.log:
            if tvt == 'train' or tvt == 'val':
                objs.insert(0, self.merged)
        return objs

    def proc_outs(self, outs, tvt, iter):
        if FLAGS.log:
            if tvt == 'train':
                self.tw.add_summary(outs[0], iter)
            elif tvt == 'val':
                self.vw.add_summary(outs[0], iter)

    def _get_model_str(self):
        li = []
        for f in [
            FLAGS.model, FLAGS.dataset, FLAGS.valid_percentage,
            FLAGS.node_feat_encoder, FLAGS.edge_feat_processor,
            FLAGS.dist_metric, FLAGS.dist_algo,
            FLAGS.sampler, FLAGS.sample_num,
            self._bool_to_str(
                FLAGS.sampler_duplicate_removal, 'samplerDuplicateRemoval'),
            FLAGS.num_layers,
            self._bool_to_str(FLAGS.dist_norm, 'normDist'),
            FLAGS.sim_kernel, FLAGS.yeta,
            FLAGS.final_act, FLAGS.loss_func, FLAGS.batch_size,
            FLAGS.learning_rate]:
            li.append(str(f))
        return '_'.join(li)

    def _log_model_info(self, logdir, sess):
        model_info_table = [["**key**", "**value**"]]
        with open(logdir + '/model_info.txt', 'w') as f:
            s = get_model_info_as_str(model_info_table)
            f.write(s)
        print(s)
        model_info_op = \
            tf.summary.text(
                'model_info', tf.convert_to_tensor(model_info_table))
        self.tw.add_summary(sess.run(model_info_op))

    # def _get_layers_str(self, FLAGS):
    #     ss = []
    #     for i in range(FLAGS.num_layers):
    #         ss.append(FLAGS.flag_values_dict()['layer_{}'.format(i)])
    #     return '_'.join(ss)

    def _bool_to_str(self, b, s):
        assert (type(b) is bool)
        if b:
            return s
        else:
            return 'NO{}'.format(s)
