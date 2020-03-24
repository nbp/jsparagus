#!/usr/bin/env python3

import io
import re
import argparse

parser = argparse.ArgumentParser(description='Profile shift actions sequences.')
parser.add_argument('filename', metavar='FILE', type=str,
                    help="File which contains the log of shift actions.")

args = parser.parse_args()
print(args.filename)
get_ints = re.compile("[-]?\d+")
token_hist = {}
with open(args.filename, 'r') as log:
    into = None
    for line in log:
        if line.startswith('shift:'): # shift: <token>
            token = line[len('shift:'):].strip()
            if token not in token_hist:
                token_hist[token] = {'count': 0}
            into = token_hist[token]
            into['count'] += 1
            pass
        elif line.startswith('shift-loop:'): # shift-loop: (src, term, dst)
            edge = line[len('shift-loop:'):].strip()
            edge = tuple(map(int, get_ints.findall(edge)))
            if edge not in into:
                into[edge] = {'count': 0}
            into = into[edge]
            into['count'] += 1

def stats(into, path):
    if len(into) == 1:
        yield into['count'], path
    else:
        for k, v in into.items():
            if k == 'count':
                continue
            for result in stats(v, path + [k]):
                yield result

def edge_str(edge):
    return "({:>3}, {:>3}, {:>4})".format(*edge)

nb_shift = 0.0
total_count = 0.0
for token, v in token_hist.items():
    for count, path in stats(v, []):
        nb_shift += len(path) * count
        total_count += count
        print("{:>7} x token {}: {}".format(count, token, ", ".join(map(edge_str, path))))
print("Average shift per tokens: {} (= {} / {})".format(nb_shift / total_count, nb_shift, total_count))


