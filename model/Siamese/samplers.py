import random
import networkx as nx


class Sampler(object):
    def __init__(self, gs, sample_num, sampler_duplicate_removal):
        self.gs = gs
        self.sample_num = sample_num
        self.sampler_duplicate_removal = sampler_duplicate_removal
        assert (len(gs) >= 2)

    def get_pair(self):
        raise NotImplementedError()

    def get_triple_for_hinge_loss(self):
        raise NotImplementedError()


class RandomSampler(Sampler):
    def __init__(self, gs, sample_num, sampler_duplicate_removal):
        super().__init__(gs, sample_num, sampler_duplicate_removal)
        self.idx = 0

    def get_pair(self):
        g1 = self.gs[self.idx]
        self.idx += 1
        if self.idx >= len(self.gs):
            random.Random(123).shuffle(self.gs)
            self.idx = 0
        g2 = self.gs[self.idx]
        return g1, g2

    def get_triple_for_hinge_loss(self):
        raise NotImplementedError()


class DistributionSampler(Sampler):
    """docstring for DistributionSampler"""

    def __init__(self, gs, sample_num, sampler_duplicate_removal, bin_size=5):

        super(DistributionSampler, self).__init__(gs, sample_num, sampler_duplicate_removal)
        densities = [nx.density(g.nxgraph) for g in self.gs]
        self.dens_list = sorted([(dense, idx) for idx, dense in enumerate(densities)])
        self.bin_size = bin_size
        self.bin_number = int(len(self.gs) / self.bin_size)
        self.bin_idx = self.shuffle_idx()
        self.item_idx = random.Random(123).randint(0, self.bin_size - 1)
        self.cur = 0

    def shuffle_idx(self):
        bins = list(range(self.bin_number))
        random.Random(123).shuffle(bins)
        if self.sample_num > 0:
            bins = bins[:2 * self.sample_num]  # assume bins number>>sample number
        return bins

    def get_pair(self):
        g1 = self.gs[self.dens_list[self.bin_idx[self.cur] * self.bin_size + self.item_idx][1]]
        g2 = self.gs[self.dens_list[self.bin_idx[self.cur + 1] * self.bin_size + self.item_idx][1]]
        self.cur += 2
        if self.cur >= len(self.bin_idx)-1:
            self.cur = 0
        self.item_idx = random.Random(123 + self.cur).randint(0, self.bin_size - 1)
        return g1, g2

    def get_triple_for_hinge_loss(self):
        raise NotImplementedError()
