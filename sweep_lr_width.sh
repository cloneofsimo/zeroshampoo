#!/bin/bash

LRS=(0.1 0.316 1 3.16 10 0.0316 0.01 0.00316 0.001 0.000316 0.0001 0.0000316 0.00001)
MODEL_WIDTHS=(64 128 512)

# # Loop over learning rates and model widths
for width in "${MODEL_WIDTHS[@]}"; do
    for lr in "${LRS[@]}"; do
        python lowprecision_shampoo.py --width $width --lr $lr --shampoo
    done
done


for width in "${MODEL_WIDTHS[@]}"; do
    for lr in "${LRS[@]}"; do
        python lowprecision_shampoo.py --width $width --lr $lr
    done
done