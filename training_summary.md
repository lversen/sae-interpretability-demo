# Training Results Summary

Date: 2025-03-28 07:43:42

## Overview

Total configurations: 27
Total runs: 27
Runs per configuration: 1
Successful: 27
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
| mnist_st_400_softmax_lr0p001_l12p5 | 1 | 0.0h | models\mnist\st\softmax\400\bs4096_lr0p001_steps30450_l12p5.pth |
| mnist_st_400_softmax_lr0p001_l110p0 | 1 | 0.0h | models\mnist\st\softmax\400\bs4096_lr0p001_steps30450_l110p0.pth |
| mnist_st_400_softmax_lr0p001 | 1 | 0.0h | models\mnist\st\softmax\400\bs4096_lr0p001_steps30450_l15p0.pth |
| mnist_st_800_softmax_lr0p001_l12p5 | 1 | 0.0h | models\mnist\st\softmax\800\bs4096_lr0p001_steps51210_l12p5.pth |
| mnist_st_800_softmax_lr0p001 | 1 | 0.0h | models\mnist\st\softmax\800\bs4096_lr0p001_steps51210_l15p0.pth |
| mnist_st_800_softmax_lr0p001_l110p0 | 1 | 0.0h | models\mnist\st\softmax\800\bs4096_lr0p001_steps51210_l110p0.pth |
| mnist_st_1600_softmax_lr0p001 | 1 | 0.0h | models\mnist\st\softmax\1600\bs4096_lr0p001_steps86145_l15p0.pth |
| mnist_st_1600_softmax_lr0p001_l12p5 | 1 | 0.0h | models\mnist\st\softmax\1600\bs4096_lr0p001_steps86145_l12p5.pth |
| mnist_st_1600_softmax_lr0p001_l110p0 | 1 | 0.0h | models\mnist\st\softmax\1600\bs4096_lr0p001_steps86145_l110p0.pth |
| mnist_st_400_relu_attention_lr0p001 | 1 | 0.0h | models\mnist\st\relu_attention\400\bs4096_lr0p001_steps30450_l15p0.pth |
| mnist_st_400_relu_attention_lr0p001_l110p0 | 1 | 0.0h | models\mnist\st\relu_attention\400\bs4096_lr0p001_steps30450_l110p0.pth |
| mnist_st_400_relu_attention_lr0p001_l12p5 | 1 | 0.0h | models\mnist\st\relu_attention\400\bs4096_lr0p001_steps30450_l12p5.pth |
| mnist_st_800_relu_attention_lr0p001_l12p5 | 1 | 1.3h | models\mnist\st\relu_attention\800\bs4096_lr0p001_steps51210_l12p5.pth |
| mnist_st_800_relu_attention_lr0p001 | 1 | 1.4h | models\mnist\st\relu_attention\800\bs4096_lr0p001_steps51210_l15p0.pth |
| mnist_st_800_relu_attention_lr0p001_l110p0 | 1 | 1.4h | models\mnist\st\relu_attention\800\bs4096_lr0p001_steps51210_l110p0.pth |
| mnist_st_1600_relu_attention_lr0p001 | 1 | 2.7h | models\mnist\st\relu_attention\1600\bs4096_lr0p001_steps86145_l15p0.pth |
| mnist_st_1600_relu_attention_lr0p001_l110p0 | 1 | 2.7h | models\mnist\st\relu_attention\1600\bs4096_lr0p001_steps86145_l110p0.pth |
| mnist_st_1600_relu_attention_lr0p001_l12p5 | 1 | 2.7h | models\mnist\st\relu_attention\1600\bs4096_lr0p001_steps86145_l12p5.pth |
| mnist_st_400_tanh_scale_shift_lr0p001 | 1 | 0.8h | models\mnist\st\tanh_scale_shift\400\bs4096_lr0p001_steps30450_l15p0.pth |
| mnist_st_400_tanh_scale_shift_lr0p001_l12p5 | 1 | 0.8h | models\mnist\st\tanh_scale_shift\400\bs4096_lr0p001_steps30450_l12p5.pth |
| mnist_st_400_tanh_scale_shift_lr0p001_l110p0 | 1 | 0.8h | models\mnist\st\tanh_scale_shift\400\bs4096_lr0p001_steps30450_l110p0.pth |
| mnist_st_800_tanh_scale_shift_lr0p001 | 1 | 1.4h | models\mnist\st\tanh_scale_shift\800\bs4096_lr0p001_steps51210_l15p0.pth |
| mnist_st_800_tanh_scale_shift_lr0p001_l12p5 | 1 | 1.4h | models\mnist\st\tanh_scale_shift\800\bs4096_lr0p001_steps51210_l12p5.pth |
| mnist_st_800_tanh_scale_shift_lr0p001_l110p0 | 1 | 1.4h | models\mnist\st\tanh_scale_shift\800\bs4096_lr0p001_steps51210_l110p0.pth |
| mnist_st_1600_tanh_scale_shift_lr0p001 | 1 | 2.7h | models\mnist\st\tanh_scale_shift\1600\bs4096_lr0p001_steps86145_l15p0.pth |
| mnist_st_1600_tanh_scale_shift_lr0p001_l12p5 | 1 | 2.7h | models\mnist\st\tanh_scale_shift\1600\bs4096_lr0p001_steps86145_l12p5.pth |
| mnist_st_1600_tanh_scale_shift_lr0p001_l110p0 | 1 | 2.7h | models\mnist\st\tanh_scale_shift\1600\bs4096_lr0p001_steps86145_l110p0.pth |

