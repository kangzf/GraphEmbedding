import random
from random import randint
import networkx as nx

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

class DistributionSampler(Sampler):
    """docstring for DistributionSampler"""
    def __init__(self, gs, sample_num, sampler_duplicate_removal):
        super(DistributionSampler, self).__init__(bin_size=5)
        self.dens_list = sorted([(nx.density[g],idx) for idx,g in enumerate(self.gs)])
        self.bin_size = bin_size
        self.bin_number = int(len(self.gs)/self.bin_size)
        self.bin_idx = self.shuffle_idx()
        self.item_idx = random.Random(123).randint(0,self.bin_size-1)
        self.cur = 0

    def shuffle_idx(self):
        bins = list(range(self.bin_number))
        random.Random(123).shuffle(bins)
        return bins[:2*self.sample_num] # assume bins number>>sample number
    
    def get_pair(self):
        g1 = self.gs[self.dens_list[self.bin_idx[self.cur]*self.bin_size+self.item_idx][1]]
        g2 = self.gs[self.dens_list[self.bin_idx[self.cur+1]*self.bin_size+self.item_idx][1]]
        self.cur += 2
        if self.cur >= len(self.bin_idx):
            self.cur = 0
        self.item_idx = random.Random(123+self.cur).randint(0,self.bin_size-1)
        return g1, g2



        

