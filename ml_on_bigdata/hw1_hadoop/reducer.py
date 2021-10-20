#!/usr/bin/env python
"""reducer.py"""

import sys

total_size = 0
total_mean = 0
total_var = 0

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input we got from mapper.py
    chunk_size, mean, var = line.split('\t')

    total_var = (total_size * total_var + chunk_size * var) / (total_size + chunk_size) + \
                total_size * chunk_size * ((total_mean - mean) / (total_size + chunk_size)) ** 2
    total_mean = (total_size * total_mean + chunk_size * mean) / (total_size + chunk_size)
    total_size += chunk_size

print(f"{total_mean}\t{total_var}")
