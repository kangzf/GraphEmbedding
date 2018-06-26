from utils_siamese import print_msec
from eval import Eval
from time import time
import numpy as np


def train_val(FLAGS, data, placeholders, dist_calculator, model, saver, sess):
    train_costs, train_times, val_costs, val_times = [], [], [], []
    for iter in range(FLAGS.iters):
        # Train.
        train_cost, train_time = run_tf(
            FLAGS, data, placeholders, dist_calculator, model, saver, sess,
            'train', iter=iter)
        train_costs.append(train_cost)
        train_times.append(train_time)

        # Validate.
        val_cost, val_time = run_tf(
            FLAGS, data, placeholders, dist_calculator, model, saver, sess,
            'val', iter=iter)
        val_costs.append(val_cost)
        val_times.append(val_time)

        # Test.
        # test_cost, test_time = run_tf('test')

        print('Iter:', '%04d' % (iter + 1),
              'train_loss=', '{:.5f}'.format(train_cost),
              'time=', print_msec(train_time),
              'val_loss=', '{:.5f}'.format(val_cost),
              'time=', print_msec(val_time))
        # 'test_loss=', '{:.5f}'.format(test_cost),
        # 'time=', '{:.5f}'.format(test_time))

        if FLAGS.early_stopping:
            if iter > FLAGS.early_stopping and \
                    val_costs[-1] > \
                    np.mean(val_costs[-(FLAGS.early_stopping + 1):-1]):
                print('Early stopping...')
                break

    print('Optimization Finished!')
    return train_costs, train_times, val_costs, val_times


def test(FLAGS, data, placeholders, dist_calculator, model, saver, sess):
    # Test.
    eval = Eval(FLAGS.dataset, FLAGS.sim_kernel, FLAGS.yeta, FLAGS.plot_results)
    m, n = data.m_n()
    test_sim_mat = np.zeros((m, n))
    test_time_mat = np.zeros((m, n))
    run_tf(FLAGS, data, placeholders, dist_calculator, model, saver, sess,
           'test', 0, 0)  # flush the pipeline
    print('i,j,time,sim,true_sim')
    for i in range(m):
        for j in range(n):
            sim_i_j, test_time = run_tf(
                FLAGS, data, placeholders, dist_calculator, model, saver, sess,
                'test', i, j)
            test_time *= 1000
            print('{},{},{:.2f}mec,{:.4f},{:.4f}'.format(
                i, j, test_time, sim_i_j,
                eval.get_true_sim(i, j, FLAGS.dist_norm)))
            # assert (0 <= sim_i_j <= 1)
            test_sim_mat[i][i] = sim_i_j
            test_time_mat[i][j] = test_time
    print('Evaluating...')
    results = eval.eval_test(FLAGS.model, test_sim_mat, test_time_mat)
    print('Results generated with {} metrics'.format(len(results)))
    return results


def run_tf(FLAGS, data, placeholders, dist_calculator, model, saver, sess, tvt,
           test_id=None, train_id=None, iter=None):
    feed_dict = data.get_feed_dict(
        FLAGS, placeholders, dist_calculator, tvt, test_id, train_id)
    if tvt == 'train':
        objs = [model.opt_op, model.loss]
        # objs = [model.pred_sim(), model.opt_op, model.loss] # TODO: figure out why it's slow
    elif tvt == 'val':
        objs = [model.loss]
    elif tvt == 'test':
        objs = [model.pred_sim_without_act()]
    else:
        raise RuntimeError('Unknown train_val_test {}'.format(tvt))
    objs = saver.proc_objs(objs, tvt)
    t = time()
    outs = sess.run(objs, feed_dict=feed_dict)
    time_rtn = time() - t
    saver.proc_outs(outs, tvt, iter)
    if tvt == 'test':
        # tf_result = sess.run([model.pred_sim()], feed_dict=feed_dict)[-1]
        # print('tf result', tf_result[-1])
        np_result = model.apply_final_act_np(outs[-1])
        # print('np result', np_result)
        outs[-1] = np_result
    return outs[-1], time_rtn
