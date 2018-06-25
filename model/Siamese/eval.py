from exp import BASELINE_MODELS, TRUE_MODEL, plot_apk, plot_mrr_mse_time
from results import load_results_as_dict, load_result


class Eval(object):
    def __init__(self, dataset, sim_kernel_name, yeta, plot_results):
        self.dataset = dataset
        if plot_results:
            self.models = BASELINE_MODELS
        else:
            self.models = [TRUE_MODEL]
        self.rs = load_results_as_dict(dataset, self.models)
        self.true_result = self.rs[TRUE_MODEL]
        self.sim_kernel_name = sim_kernel_name
        self.yeta = yeta
        self.results = {}
        self.plot_results = plot_results

    def get_true_sim(self, query_id, train_id, norm):
        sim_mat = self.true_result.sim_mat(
            self.sim_kernel_name, self.yeta, norm)
        return sim_mat[query_id][train_id]

    def eval_test(self, cur_model, sim_mat, time_mat):
        models = self.models + [cur_model]
        self.rs[cur_model] = load_result(
            self.dataset, cur_model, sim_mat=sim_mat, time_mat=time_mat)
        norms = [True, False]
        d = plot_apk(
            self.dataset, models, self.rs, self.true_result, 'ap@k', norms,
            self.plot_results)
        self.results.update(d)
        metrics = ['mrr', 'mse']
        if time_mat is not None:
            metrics.append('time')
        for metric in metrics:
            d = plot_mrr_mse_time(
                self.dataset, models, self.rs, self.true_result, metric,
                norms, self.sim_kernel_name, self.yeta, self.plot_results)
            self.results.update(d)
        return self.results
