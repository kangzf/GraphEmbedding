import random


class Sampler(object):
    def __init__(self, gs, sample_num, sampler_duplicate_removal):
        self.gs = gs
        self.sample_num = sample_num
        self.sampler_duplicate_removal = sampler_duplicate_removal
        assert (len(gs) >= 2)

    def get_pair(self):
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
