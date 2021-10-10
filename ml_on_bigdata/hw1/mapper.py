#!/usr/bin/env python
"""mapper.py"""

import sys
from csv import reader

PRICE_IX = 9

chunk_size = 0
mean = 0
var = 0

# input comes from STDIN (standard input)
for line in reader(sys.stdin):
    try:
        price = float(line[PRICE_IX])
    except ValueError:
        # price is not present in current line
        continue

    var = chunk_size * var / (chunk_size + 1) + chunk_size * ((mean - price) / (chunk_size + 1)) ** 2
    mean = (chunk_size * mean + price) / (chunk_size + 1)
    chunk_size += 1

print(f"{chunk_size}\t{mean}\t{var}")
