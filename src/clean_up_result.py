#!/usr/bin/env python3
from utils import get_result_path, sorted_nicely, exec
from glob import glob
from os.path import basename, splitext
import numpy as np

# f = '/aids10k/csv/*.csv'
f = '/aids10k/ged/*.npy'

def clean_up():
    rp = get_result_path()
    for file in sorted_nicely(glob('{}/{}'.format(rp, f))):
        bnf = basename(file)
        print_info(file, bnf)
        t = prompt('Delete? [y/n]', ['y', 'n'])
        if t == 'y':
            exec('rm -rf {}'.format(file))
        elif t == 'n':
            print('Skip')
        else:
            assert (False)
    print('Done')

def print_info(file, bnf):
    ext = splitext(bnf)[1]
    if ext == '.csv':
        num_lines = sum(1 for _ in open(file))
        print('{} has {} lines'.format(bnf, num_lines))
    elif ext == '.npy':
        print('{} shape {}'.format(bnf, np.load(file).shape))
    else:
        print(ext)


def prompt(str, options):
    while True:
        t = input(str)
        if t in options:
            return t


clean_up()
