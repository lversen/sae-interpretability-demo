# Training Results Summary

Date: 2025-03-25 12:16:12

## Overview

Total configurations: 18
Total runs: 18
Runs per configuration: 1
Successful: 18
Skipped: 0
Failed: 0

## Hierarchical Organization

Models are organized in the following structure:
```
models/
  |-- [dataset]/
      |-- sae/
      |   |-- [activation_function]/
      |   |   |-- [feature_dimension]/
      |   |       |-- bs{batch}_lr{lr}_steps{steps}.pth         # Run 1
      |   |       |-- bs{batch}_lr{lr}_steps{steps}_2.pth       # Run 2
      |   |       |-- bs{batch}_lr{lr}_steps{steps}_3.pth       # Run 3
      |-- st/
          |-- [attention_function]/
              |-- [feature_dimension]/
                  |-- bs{batch}_lr{lr}_steps{steps}.pth         # Run 1
                  |-- bs{batch}_lr{lr}_steps{steps}_2.pth       # Run 2
                  |-- bs{batch}_lr{lr}_steps{steps}_3.pth       # Run 3
```

## Successful Models

| Configuration | Run | Training Time | Path |
|--------------|-----|---------------|------|
| mnist_st_20_softmax_lr0p001_l110p0 | 1 | 0.0h | models\mnist\st\softmax\20\bs4096_lr0p001_steps6000_l110p0.pth |
| mnist_st_20_softmax_l110p0 | 1 | 0.0h | models\mnist\st\softmax\20\bs4096_lr5e-05_steps6000_l110p0.pth |
| mnist_st_50_softmax_lr0p001_l110p0 | 1 | 0.0h | models\mnist\st\softmax\50\bs4096_lr0p001_steps6390_l110p0.pth |
| mnist_st_50_softmax_l110p0 | 1 | 0.0h | models\mnist\st\softmax\50\bs4096_lr5e-05_steps6390_l110p0.pth |
| mnist_st_100_softmax_lr0p001_l110p0 | 1 | 0.0h | models\mnist\st\softmax\100\bs4096_lr0p001_steps10755_l110p0.pth |
| mnist_st_100_softmax_l110p0 | 1 | 0.3h | models\mnist\st\softmax\100\bs4096_lr5e-05_steps10755_l110p0.pth |
| mnist_st_20_relu_attention_lr0p001_l110p0 | 1 | 0.1h | models\mnist\st\relu_attention\20\bs4096_lr0p001_steps6000_l110p0.pth |
| mnist_st_20_relu_attention_l110p0 | 1 | 0.1h | models\mnist\st\relu_attention\20\bs4096_lr5e-05_steps6000_l110p0.pth |
| mnist_st_50_relu_attention_lr0p001_l110p0 | 1 | 0.2h | models\mnist\st\relu_attention\50\bs4096_lr0p001_steps6390_l110p0.pth |
| mnist_st_50_relu_attention_l110p0 | 1 | 0.1h | models\mnist\st\relu_attention\50\bs4096_lr5e-05_steps6390_l110p0.pth |
| mnist_st_100_relu_attention_lr0p001_l110p0 | 1 | 0.2h | models\mnist\st\relu_attention\100\bs4096_lr0p001_steps10755_l110p0.pth |
| mnist_st_100_relu_attention_l110p0 | 1 | 0.3h | models\mnist\st\relu_attention\100\bs4096_lr5e-05_steps10755_l110p0.pth |
| mnist_st_20_tanh_scale_shift_lr0p001_l110p0 | 1 | 0.1h | models\mnist\st\tanh_scale_shift\20\bs4096_lr0p001_steps6000_l110p0.pth |
| mnist_st_20_tanh_scale_shift_l110p0 | 1 | 0.1h | models\mnist\st\tanh_scale_shift\20\bs4096_lr5e-05_steps6000_l110p0.pth |
| mnist_st_50_tanh_scale_shift_lr0p001_l110p0 | 1 | 0.2h | models\mnist\st\tanh_scale_shift\50\bs4096_lr0p001_steps6390_l110p0.pth |
| mnist_st_50_tanh_scale_shift_l110p0 | 1 | 0.2h | models\mnist\st\tanh_scale_shift\50\bs4096_lr5e-05_steps6390_l110p0.pth |
| mnist_st_100_tanh_scale_shift_lr0p001_l110p0 | 1 | 0.3h | models\mnist\st\tanh_scale_shift\100\bs4096_lr0p001_steps10755_l110p0.pth |
| mnist_st_100_tanh_scale_shift_l110p0 | 1 | 0.3h | models\mnist\st\tanh_scale_shift\100\bs4096_lr5e-05_steps10755_l110p0.pth |

