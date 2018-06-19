import sys
from os.path import dirname, abspath

sys.path.insert(0, "{}/../src".format(dirname(dirname(abspath(__file__)))))
from exp import BASELINE_MODELS, TRUE_MODEL, plot_apk, plot_mrr_mse
from results import load_results_as_dict


def check_flags(FLAGS):
    assert (FLAGS.sample_num >= -1)
    assert (FLAGS.yeta >= 0)
    # TODO: finish.


def eval_test(dataset, cur_model, sim_mat, time_mat, sim_kernel, yeta):
    models = BASELINE_MODELS + [cur_model]
    norms = [True, False]
    rs = load_results_as_dict(dataset, models, \
                              sim_mat=sim_mat, time_mat=time_mat)
    true_result = rs[TRUE_MODEL]
    plot_apk(dataset, models, rs, true_result, 'ap@k', norms)
    plot_mrr_mse(dataset, models, rs, true_result, 'mse', norms, \
                 sim_kernel, yeta)