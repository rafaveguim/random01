import argparse, sys
from test import eval_list
from itertools import islice
from pprint import pprint

def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))

def options():
    parser = argparse.ArgumentParser(description='Classifies strings '
        'as random/non-random.')
    parser.add_argument('strings',
        nargs='?',
        type=argparse.FileType('r'),
        default=sys.stdin)

    return parser.parse_args()

if __name__ == '__main__':
    opts = options()
    strings = opts.strings

    for batch in split_every(1000, (s.rstrip() for s in strings)):
        for string, label in eval_list(batch):
            print("{}\t{}".format(string, label))
