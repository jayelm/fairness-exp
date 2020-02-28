#!/usr/bin/env bash

for max_length in 1 2 3 4; do
    for thresh in 0.04 0.20 0.50; do
        python code/main.py analyze --max_formula_length $max_length --neuron_threshold $thresh --save_analysis analysis/data/"$max_length"_"$thresh".csv
    done
done
