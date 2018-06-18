from random import shuffle


class Sampler(object):
    def __init__(self, gs):
        self.gs = gs
        assert (len(gs) >= 2)

    def get_pair(self):
        raise NotImplementedError()


class RandomSampler(Sampler):
    def __init__(self, gs):
        super().__init__(gs)
        self.idx = 0

    def get_pair(self):
        g1 = self.gs[self.idx]
        self.idx += 1
        if self.idx >= len(self.gs):
            shuffle(self.gs)
            self.idx = 0
        g2 = self.gs[self.idx]
        return g1, g2
