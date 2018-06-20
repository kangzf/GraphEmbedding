import numpy as np
import tensorflow as tf


class SimilarityKernel(object):
    def dist_to_sim(self, dist, max_dist):
        raise NotImplementedError()

    def dist_to_sim_tf(self, dist, max_dist):
        raise NotImplementedError()


class LinearKernel:
    def dist_to_sim(self, dist, max_dist):
        return self.__d_to_s(dist, max_dist)

    def dist_to_sim_tf(self, dist, max_dist):
        return self.__d_to_s(dist, max_dist)

    def __d_to_s(self, dist, max_dist):
        return 1 - dist / max_dist


class GaussianKernel(SimilarityKernel):
    def __init__(self, yeta):
        self.yeta = yeta
        pass

    def dist_to_sim(self, dist, *unused):
        return np.exp(-self.yeta * np.square(dist))

    def dist_to_sim_tf(self, dist, *unuse):
        return tf.exp(-self.yeta * tf.square(dist))


class BinaryKernel(SimilarityKernel):
    def __init__(self, threshold):
        self.threshold = threshold

    # TODO: simplify ranking into binary classification


def create_sim_kernel(kernel_name, yeta=None):
    if kernel_name == 'linear':
        return LinearKernel()
    elif kernel_name == 'gaussian':
        return GaussianKernel(yeta)
    else:
        raise RuntimeError('Unknown sim kernel {}'.format(kernel_name))
