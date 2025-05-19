# Training Results Summary

Date: 2025-05-16 11:53:48

## Overview

Total configurations: 252
Total runs: 252
Runs per configuration: 1
Successful: 252
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
| mnist_st_20_softmax_oldst | 1 | 0.0h | models\mnist\st\softmax\20\bs4096_lr5e-05_steps6000_l15p0_oldst.pth |
| mnist_st_20_softmax_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\20\bs4096_lr5e-05_steps6000_l110p0_oldst.pth |
| mnist_st_20_softmax_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\20\bs4096_lr5e-05_steps6000_l12p5_oldst.pth |
| mnist_st_20_softmax_lr0p001_oldst | 1 | 0.0h | models\mnist\st\softmax\20\bs4096_lr0p001_steps6000_l15p0_oldst.pth |
| mnist_st_20_softmax_lr0p001_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\20\bs4096_lr0p001_steps6000_l12p5_oldst.pth |
| mnist_st_20_softmax_lr0p001_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\20\bs4096_lr0p001_steps6000_l110p0_oldst.pth |
| mnist_st_50_softmax_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\50\bs4096_lr5e-05_steps6390_l12p5_oldst.pth |
| mnist_st_50_softmax_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\50\bs4096_lr5e-05_steps6390_l110p0_oldst.pth |
| mnist_st_50_softmax_oldst | 1 | 0.0h | models\mnist\st\softmax\50\bs4096_lr5e-05_steps6390_l15p0_oldst.pth |
| mnist_st_50_softmax_lr0p001_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\50\bs4096_lr0p001_steps6390_l12p5_oldst.pth |
| mnist_st_50_softmax_lr0p001_oldst | 1 | 0.0h | models\mnist\st\softmax\50\bs4096_lr0p001_steps6390_l15p0_oldst.pth |
| mnist_st_50_softmax_lr0p001_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\50\bs4096_lr0p001_steps6390_l110p0_oldst.pth |
| mnist_st_100_softmax_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\100\bs4096_lr5e-05_steps10755_l12p5_oldst.pth |
| mnist_st_100_softmax_oldst | 1 | 0.0h | models\mnist\st\softmax\100\bs4096_lr5e-05_steps10755_l15p0_oldst.pth |
| mnist_st_100_softmax_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\100\bs4096_lr5e-05_steps10755_l110p0_oldst.pth |
| mnist_st_100_softmax_lr0p001_oldst | 1 | 0.0h | models\mnist\st\softmax\100\bs4096_lr0p001_steps10755_l15p0_oldst.pth |
| mnist_st_100_softmax_lr0p001_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\100\bs4096_lr0p001_steps10755_l12p5_oldst.pth |
| mnist_st_100_softmax_lr0p001_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\100\bs4096_lr0p001_steps10755_l110p0_oldst.pth |
| mnist_st_200_softmax_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\200\bs4096_lr5e-05_steps18105_l12p5_oldst.pth |
| mnist_st_200_softmax_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\200\bs4096_lr5e-05_steps18105_l110p0_oldst.pth |
| mnist_st_200_softmax_oldst | 1 | 0.0h | models\mnist\st\softmax\200\bs4096_lr5e-05_steps18105_l15p0_oldst.pth |
| mnist_st_200_softmax_lr0p001_oldst | 1 | 0.0h | models\mnist\st\softmax\200\bs4096_lr0p001_steps18105_l15p0_oldst.pth |
| mnist_st_200_softmax_lr0p001_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\200\bs4096_lr0p001_steps18105_l12p5_oldst.pth |
| mnist_st_200_softmax_lr0p001_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\200\bs4096_lr0p001_steps18105_l110p0_oldst.pth |
| mnist_st_400_softmax_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\400\bs4096_lr5e-05_steps30450_l12p5_oldst.pth |
| mnist_st_400_softmax_oldst | 1 | 0.0h | models\mnist\st\softmax\400\bs4096_lr5e-05_steps30450_l15p0_oldst.pth |
| mnist_st_400_softmax_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\400\bs4096_lr5e-05_steps30450_l110p0_oldst.pth |
| mnist_st_400_softmax_lr0p001_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\400\bs4096_lr0p001_steps30450_l12p5_oldst.pth |
| mnist_st_400_softmax_lr0p001_oldst | 1 | 0.0h | models\mnist\st\softmax\400\bs4096_lr0p001_steps30450_l15p0_oldst.pth |
| mnist_st_400_softmax_lr0p001_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\400\bs4096_lr0p001_steps30450_l110p0_oldst.pth |
| mnist_st_800_softmax_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\800\bs4096_lr5e-05_steps51210_l12p5_oldst.pth |
| mnist_st_800_softmax_oldst | 1 | 0.0h | models\mnist\st\softmax\800\bs4096_lr5e-05_steps51210_l15p0_oldst.pth |
| mnist_st_800_softmax_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\800\bs4096_lr5e-05_steps51210_l110p0_oldst.pth |
| mnist_st_800_softmax_lr0p001_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\800\bs4096_lr0p001_steps51210_l12p5_oldst.pth |
| mnist_st_800_softmax_lr0p001_oldst | 1 | 0.0h | models\mnist\st\softmax\800\bs4096_lr0p001_steps51210_l15p0_oldst.pth |
| mnist_st_800_softmax_lr0p001_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\800\bs4096_lr0p001_steps51210_l110p0_oldst.pth |
| mnist_st_1600_softmax_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\1600\bs4096_lr5e-05_steps86145_l12p5_oldst.pth |
| mnist_st_1600_softmax_oldst | 1 | 0.0h | models\mnist\st\softmax\1600\bs4096_lr5e-05_steps86145_l15p0_oldst.pth |
| mnist_st_1600_softmax_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\1600\bs4096_lr5e-05_steps86145_l110p0_oldst.pth |
| mnist_st_1600_softmax_lr0p001_l12p5_oldst | 1 | 0.0h | models\mnist\st\softmax\1600\bs4096_lr0p001_steps86145_l12p5_oldst.pth |
| mnist_st_1600_softmax_lr0p001_oldst | 1 | 0.0h | models\mnist\st\softmax\1600\bs4096_lr0p001_steps86145_l15p0_oldst.pth |
| mnist_st_1600_softmax_lr0p001_l110p0_oldst | 1 | 0.0h | models\mnist\st\softmax\1600\bs4096_lr0p001_steps86145_l110p0_oldst.pth |
| mnist_st_20_relu_softmax_l12p5_oldst | 1 | 0.1h | models\mnist\st\relu_softmax\20\bs4096_lr5e-05_steps6000_l12p5_oldst.pth |
| mnist_st_20_relu_softmax_l110p0_oldst | 1 | 0.1h | models\mnist\st\relu_softmax\20\bs4096_lr5e-05_steps6000_l110p0_oldst.pth |
| mnist_st_20_relu_softmax_oldst | 1 | 0.1h | models\mnist\st\relu_softmax\20\bs4096_lr5e-05_steps6000_l15p0_oldst.pth |
| mnist_st_20_relu_softmax_lr0p001_oldst | 1 | 0.1h | models\mnist\st\relu_softmax\20\bs4096_lr0p001_steps6000_l15p0_oldst.pth |
| mnist_st_20_relu_softmax_lr0p001_l12p5_oldst | 1 | 0.1h | models\mnist\st\relu_softmax\20\bs4096_lr0p001_steps6000_l12p5_oldst.pth |
| mnist_st_20_relu_softmax_lr0p001_l110p0_oldst | 1 | 0.1h | models\mnist\st\relu_softmax\20\bs4096_lr0p001_steps6000_l110p0_oldst.pth |
| mnist_st_50_relu_softmax_l12p5_oldst | 1 | 0.1h | models\mnist\st\relu_softmax\50\bs4096_lr5e-05_steps6390_l12p5_oldst.pth |
| mnist_st_50_relu_softmax_l110p0_oldst | 1 | 0.1h | models\mnist\st\relu_softmax\50\bs4096_lr5e-05_steps6390_l110p0_oldst.pth |
| mnist_st_50_relu_softmax_oldst | 1 | 0.1h | models\mnist\st\relu_softmax\50\bs4096_lr5e-05_steps6390_l15p0_oldst.pth |
| mnist_st_50_relu_softmax_lr0p001_oldst | 1 | 0.1h | models\mnist\st\relu_softmax\50\bs4096_lr0p001_steps6390_l15p0_oldst.pth |
| mnist_st_50_relu_softmax_lr0p001_l110p0_oldst | 1 | 0.1h | models\mnist\st\relu_softmax\50\bs4096_lr0p001_steps6390_l110p0_oldst.pth |
| mnist_st_50_relu_softmax_lr0p001_l12p5_oldst | 1 | 0.1h | models\mnist\st\relu_softmax\50\bs4096_lr0p001_steps6390_l12p5_oldst.pth |
| mnist_st_100_relu_softmax_oldst | 1 | 0.2h | models\mnist\st\relu_softmax\100\bs4096_lr5e-05_steps10755_l15p0_oldst.pth |
| mnist_st_100_relu_softmax_l12p5_oldst | 1 | 0.2h | models\mnist\st\relu_softmax\100\bs4096_lr5e-05_steps10755_l12p5_oldst.pth |
| mnist_st_100_relu_softmax_l110p0_oldst | 1 | 0.2h | models\mnist\st\relu_softmax\100\bs4096_lr5e-05_steps10755_l110p0_oldst.pth |
| mnist_st_100_relu_softmax_lr0p001_oldst | 1 | 0.2h | models\mnist\st\relu_softmax\100\bs4096_lr0p001_steps10755_l15p0_oldst.pth |
| mnist_st_100_relu_softmax_lr0p001_l110p0_oldst | 1 | 0.2h | models\mnist\st\relu_softmax\100\bs4096_lr0p001_steps10755_l110p0_oldst.pth |
| mnist_st_100_relu_softmax_lr0p001_l12p5_oldst | 1 | 0.2h | models\mnist\st\relu_softmax\100\bs4096_lr0p001_steps10755_l12p5_oldst.pth |
| mnist_st_200_relu_softmax_l12p5_oldst | 1 | 0.4h | models\mnist\st\relu_softmax\200\bs4096_lr5e-05_steps18105_l12p5_oldst.pth |
| mnist_st_200_relu_softmax_oldst | 1 | 0.4h | models\mnist\st\relu_softmax\200\bs4096_lr5e-05_steps18105_l15p0_oldst.pth |
| mnist_st_200_relu_softmax_l110p0_oldst | 1 | 0.4h | models\mnist\st\relu_softmax\200\bs4096_lr5e-05_steps18105_l110p0_oldst.pth |
| mnist_st_200_relu_softmax_lr0p001_oldst | 1 | 0.4h | models\mnist\st\relu_softmax\200\bs4096_lr0p001_steps18105_l15p0_oldst.pth |
| mnist_st_200_relu_softmax_lr0p001_l12p5_oldst | 1 | 0.4h | models\mnist\st\relu_softmax\200\bs4096_lr0p001_steps18105_l12p5_oldst.pth |
| mnist_st_200_relu_softmax_lr0p001_l110p0_oldst | 1 | 0.4h | models\mnist\st\relu_softmax\200\bs4096_lr0p001_steps18105_l110p0_oldst.pth |
| mnist_st_400_relu_softmax_l12p5_oldst | 1 | 0.6h | models\mnist\st\relu_softmax\400\bs4096_lr5e-05_steps30450_l12p5_oldst.pth |
| mnist_st_400_relu_softmax_l110p0_oldst | 1 | 0.6h | models\mnist\st\relu_softmax\400\bs4096_lr5e-05_steps30450_l110p0_oldst.pth |
| mnist_st_400_relu_softmax_oldst | 1 | 0.6h | models\mnist\st\relu_softmax\400\bs4096_lr5e-05_steps30450_l15p0_oldst.pth |
| mnist_st_400_relu_softmax_lr0p001_l12p5_oldst | 1 | 0.6h | models\mnist\st\relu_softmax\400\bs4096_lr0p001_steps30450_l12p5_oldst.pth |
| mnist_st_400_relu_softmax_lr0p001_l110p0_oldst | 1 | 0.6h | models\mnist\st\relu_softmax\400\bs4096_lr0p001_steps30450_l110p0_oldst.pth |
| mnist_st_400_relu_softmax_lr0p001_oldst | 1 | 0.6h | models\mnist\st\relu_softmax\400\bs4096_lr0p001_steps30450_l15p0_oldst.pth |
| mnist_st_800_relu_softmax_l110p0_oldst | 1 | 1.1h | models\mnist\st\relu_softmax\800\bs4096_lr5e-05_steps51210_l110p0_oldst.pth |
| mnist_st_800_relu_softmax_oldst | 1 | 1.1h | models\mnist\st\relu_softmax\800\bs4096_lr5e-05_steps51210_l15p0_oldst.pth |
| mnist_st_800_relu_softmax_l12p5_oldst | 1 | 1.1h | models\mnist\st\relu_softmax\800\bs4096_lr5e-05_steps51210_l12p5_oldst.pth |
| mnist_st_800_relu_softmax_lr0p001_l110p0_oldst | 1 | 1.1h | models\mnist\st\relu_softmax\800\bs4096_lr0p001_steps51210_l110p0_oldst.pth |
| mnist_st_800_relu_softmax_lr0p001_oldst | 1 | 1.1h | models\mnist\st\relu_softmax\800\bs4096_lr0p001_steps51210_l15p0_oldst.pth |
| mnist_st_800_relu_softmax_lr0p001_l12p5_oldst | 1 | 1.1h | models\mnist\st\relu_softmax\800\bs4096_lr0p001_steps51210_l12p5_oldst.pth |
| mnist_st_1600_relu_softmax_l110p0_oldst | 1 | 2.4h | models\mnist\st\relu_softmax\1600\bs4096_lr5e-05_steps86145_l110p0_oldst.pth |
| mnist_st_1600_relu_softmax_l12p5_oldst | 1 | 2.4h | models\mnist\st\relu_softmax\1600\bs4096_lr5e-05_steps86145_l12p5_oldst.pth |
| mnist_st_1600_relu_softmax_oldst | 1 | 2.4h | models\mnist\st\relu_softmax\1600\bs4096_lr5e-05_steps86145_l15p0_oldst.pth |
| mnist_st_1600_relu_softmax_lr0p001_l12p5_oldst | 1 | 2.4h | models\mnist\st\relu_softmax\1600\bs4096_lr0p001_steps86145_l12p5_oldst.pth |
| mnist_st_1600_relu_softmax_lr0p001_l110p0_oldst | 1 | 2.4h | models\mnist\st\relu_softmax\1600\bs4096_lr0p001_steps86145_l110p0_oldst.pth |
| mnist_st_1600_relu_softmax_lr0p001_oldst | 1 | 2.4h | models\mnist\st\relu_softmax\1600\bs4096_lr0p001_steps86145_l15p0_oldst.pth |
| mnist_st_20_tanh_scale_shift_l12p5_oldst | 1 | 0.1h | models\mnist\st\tanh_scale_shift\20\bs4096_lr5e-05_steps6000_l12p5_oldst.pth |
| mnist_st_20_tanh_scale_shift_oldst | 1 | 0.1h | models\mnist\st\tanh_scale_shift\20\bs4096_lr5e-05_steps6000_l15p0_oldst.pth |
| mnist_st_20_tanh_scale_shift_l110p0_oldst | 1 | 0.1h | models\mnist\st\tanh_scale_shift\20\bs4096_lr5e-05_steps6000_l110p0_oldst.pth |
| mnist_st_20_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 0.1h | models\mnist\st\tanh_scale_shift\20\bs4096_lr0p001_steps6000_l12p5_oldst.pth |
| mnist_st_20_tanh_scale_shift_lr0p001_oldst | 1 | 0.1h | models\mnist\st\tanh_scale_shift\20\bs4096_lr0p001_steps6000_l15p0_oldst.pth |
| mnist_st_20_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 0.1h | models\mnist\st\tanh_scale_shift\20\bs4096_lr0p001_steps6000_l110p0_oldst.pth |
| mnist_st_50_tanh_scale_shift_l12p5_oldst | 1 | 0.1h | models\mnist\st\tanh_scale_shift\50\bs4096_lr5e-05_steps6390_l12p5_oldst.pth |
| mnist_st_50_tanh_scale_shift_oldst | 1 | 0.1h | models\mnist\st\tanh_scale_shift\50\bs4096_lr5e-05_steps6390_l15p0_oldst.pth |
| mnist_st_50_tanh_scale_shift_l110p0_oldst | 1 | 0.1h | models\mnist\st\tanh_scale_shift\50\bs4096_lr5e-05_steps6390_l110p0_oldst.pth |
| mnist_st_50_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 0.1h | models\mnist\st\tanh_scale_shift\50\bs4096_lr0p001_steps6390_l12p5_oldst.pth |
| mnist_st_50_tanh_scale_shift_lr0p001_oldst | 1 | 0.1h | models\mnist\st\tanh_scale_shift\50\bs4096_lr0p001_steps6390_l15p0_oldst.pth |
| mnist_st_50_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 0.1h | models\mnist\st\tanh_scale_shift\50\bs4096_lr0p001_steps6390_l110p0_oldst.pth |
| mnist_st_100_tanh_scale_shift_l12p5_oldst | 1 | 0.2h | models\mnist\st\tanh_scale_shift\100\bs4096_lr5e-05_steps10755_l12p5_oldst.pth |
| mnist_st_100_tanh_scale_shift_oldst | 1 | 0.2h | models\mnist\st\tanh_scale_shift\100\bs4096_lr5e-05_steps10755_l15p0_oldst.pth |
| mnist_st_100_tanh_scale_shift_l110p0_oldst | 1 | 0.2h | models\mnist\st\tanh_scale_shift\100\bs4096_lr5e-05_steps10755_l110p0_oldst.pth |
| mnist_st_100_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 0.2h | models\mnist\st\tanh_scale_shift\100\bs4096_lr0p001_steps10755_l12p5_oldst.pth |
| mnist_st_100_tanh_scale_shift_lr0p001_oldst | 1 | 0.2h | models\mnist\st\tanh_scale_shift\100\bs4096_lr0p001_steps10755_l15p0_oldst.pth |
| mnist_st_100_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 0.2h | models\mnist\st\tanh_scale_shift\100\bs4096_lr0p001_steps10755_l110p0_oldst.pth |
| mnist_st_200_tanh_scale_shift_oldst | 1 | 0.4h | models\mnist\st\tanh_scale_shift\200\bs4096_lr5e-05_steps18105_l15p0_oldst.pth |
| mnist_st_200_tanh_scale_shift_l12p5_oldst | 1 | 0.4h | models\mnist\st\tanh_scale_shift\200\bs4096_lr5e-05_steps18105_l12p5_oldst.pth |
| mnist_st_200_tanh_scale_shift_l110p0_oldst | 1 | 0.4h | models\mnist\st\tanh_scale_shift\200\bs4096_lr5e-05_steps18105_l110p0_oldst.pth |
| mnist_st_200_tanh_scale_shift_lr0p001_oldst | 1 | 0.4h | models\mnist\st\tanh_scale_shift\200\bs4096_lr0p001_steps18105_l15p0_oldst.pth |
| mnist_st_200_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 0.4h | models\mnist\st\tanh_scale_shift\200\bs4096_lr0p001_steps18105_l12p5_oldst.pth |
| mnist_st_200_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 0.4h | models\mnist\st\tanh_scale_shift\200\bs4096_lr0p001_steps18105_l110p0_oldst.pth |
| mnist_st_400_tanh_scale_shift_oldst | 1 | 0.6h | models\mnist\st\tanh_scale_shift\400\bs4096_lr5e-05_steps30450_l15p0_oldst.pth |
| mnist_st_400_tanh_scale_shift_l12p5_oldst | 1 | 0.6h | models\mnist\st\tanh_scale_shift\400\bs4096_lr5e-05_steps30450_l12p5_oldst.pth |
| mnist_st_400_tanh_scale_shift_l110p0_oldst | 1 | 0.6h | models\mnist\st\tanh_scale_shift\400\bs4096_lr5e-05_steps30450_l110p0_oldst.pth |
| mnist_st_400_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 0.6h | models\mnist\st\tanh_scale_shift\400\bs4096_lr0p001_steps30450_l12p5_oldst.pth |
| mnist_st_400_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 0.6h | models\mnist\st\tanh_scale_shift\400\bs4096_lr0p001_steps30450_l110p0_oldst.pth |
| mnist_st_400_tanh_scale_shift_lr0p001_oldst | 1 | 0.6h | models\mnist\st\tanh_scale_shift\400\bs4096_lr0p001_steps30450_l15p0_oldst.pth |
| mnist_st_800_tanh_scale_shift_l12p5_oldst | 1 | 1.1h | models\mnist\st\tanh_scale_shift\800\bs4096_lr5e-05_steps51210_l12p5_oldst.pth |
| mnist_st_800_tanh_scale_shift_l110p0_oldst | 1 | 1.1h | models\mnist\st\tanh_scale_shift\800\bs4096_lr5e-05_steps51210_l110p0_oldst.pth |
| mnist_st_800_tanh_scale_shift_oldst | 1 | 1.1h | models\mnist\st\tanh_scale_shift\800\bs4096_lr5e-05_steps51210_l15p0_oldst.pth |
| mnist_st_800_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 1.1h | models\mnist\st\tanh_scale_shift\800\bs4096_lr0p001_steps51210_l12p5_oldst.pth |
| mnist_st_800_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 1.1h | models\mnist\st\tanh_scale_shift\800\bs4096_lr0p001_steps51210_l110p0_oldst.pth |
| mnist_st_800_tanh_scale_shift_lr0p001_oldst | 1 | 1.1h | models\mnist\st\tanh_scale_shift\800\bs4096_lr0p001_steps51210_l15p0_oldst.pth |
| mnist_st_1600_tanh_scale_shift_l110p0_oldst | 1 | 2.3h | models\mnist\st\tanh_scale_shift\1600\bs4096_lr5e-05_steps86145_l110p0_oldst.pth |
| mnist_st_1600_tanh_scale_shift_oldst | 1 | 2.3h | models\mnist\st\tanh_scale_shift\1600\bs4096_lr5e-05_steps86145_l15p0_oldst.pth |
| mnist_st_1600_tanh_scale_shift_l12p5_oldst | 1 | 2.3h | models\mnist\st\tanh_scale_shift\1600\bs4096_lr5e-05_steps86145_l12p5_oldst.pth |
| mnist_st_1600_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 2.3h | models\mnist\st\tanh_scale_shift\1600\bs4096_lr0p001_steps86145_l110p0_oldst.pth |
| mnist_st_1600_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 2.3h | models\mnist\st\tanh_scale_shift\1600\bs4096_lr0p001_steps86145_l12p5_oldst.pth |
| mnist_st_1600_tanh_scale_shift_lr0p001_oldst | 1 | 2.3h | models\mnist\st\tanh_scale_shift\1600\bs4096_lr0p001_steps86145_l15p0_oldst.pth |
| fashion_mnist_st_20_softmax_l12p5_oldst | 1 | 0.1h | models\fashion_mnist\st\softmax\20\bs4096_lr5e-05_steps6000_l12p5_oldst.pth |
| fashion_mnist_st_20_softmax_l110p0_oldst | 1 | 0.1h | models\fashion_mnist\st\softmax\20\bs4096_lr5e-05_steps6000_l110p0_oldst.pth |
| fashion_mnist_st_20_softmax_oldst | 1 | 0.1h | models\fashion_mnist\st\softmax\20\bs4096_lr5e-05_steps6000_l15p0_oldst.pth |
| fashion_mnist_st_20_softmax_lr0p001_l12p5_oldst | 1 | 0.1h | models\fashion_mnist\st\softmax\20\bs4096_lr0p001_steps6000_l12p5_oldst.pth |
| fashion_mnist_st_20_softmax_lr0p001_oldst | 1 | 0.1h | models\fashion_mnist\st\softmax\20\bs4096_lr0p001_steps6000_l15p0_oldst.pth |
| fashion_mnist_st_20_softmax_lr0p001_l110p0_oldst | 1 | 0.1h | models\fashion_mnist\st\softmax\20\bs4096_lr0p001_steps6000_l110p0_oldst.pth |
| fashion_mnist_st_50_softmax_l12p5_oldst | 1 | 0.1h | models\fashion_mnist\st\softmax\50\bs4096_lr5e-05_steps6390_l12p5_oldst.pth |
| fashion_mnist_st_50_softmax_oldst | 1 | 0.1h | models\fashion_mnist\st\softmax\50\bs4096_lr5e-05_steps6390_l15p0_oldst.pth |
| fashion_mnist_st_50_softmax_l110p0_oldst | 1 | 0.1h | models\fashion_mnist\st\softmax\50\bs4096_lr5e-05_steps6390_l110p0_oldst.pth |
| fashion_mnist_st_50_softmax_lr0p001_l12p5_oldst | 1 | 0.1h | models\fashion_mnist\st\softmax\50\bs4096_lr0p001_steps6390_l12p5_oldst.pth |
| fashion_mnist_st_50_softmax_lr0p001_oldst | 1 | 0.1h | models\fashion_mnist\st\softmax\50\bs4096_lr0p001_steps6390_l15p0_oldst.pth |
| fashion_mnist_st_50_softmax_lr0p001_l110p0_oldst | 1 | 0.1h | models\fashion_mnist\st\softmax\50\bs4096_lr0p001_steps6390_l110p0_oldst.pth |
| fashion_mnist_st_100_softmax_l12p5_oldst | 1 | 0.2h | models\fashion_mnist\st\softmax\100\bs4096_lr5e-05_steps10755_l12p5_oldst.pth |
| fashion_mnist_st_100_softmax_oldst | 1 | 0.2h | models\fashion_mnist\st\softmax\100\bs4096_lr5e-05_steps10755_l15p0_oldst.pth |
| fashion_mnist_st_100_softmax_l110p0_oldst | 1 | 0.2h | models\fashion_mnist\st\softmax\100\bs4096_lr5e-05_steps10755_l110p0_oldst.pth |
| fashion_mnist_st_100_softmax_lr0p001_l12p5_oldst | 1 | 0.2h | models\fashion_mnist\st\softmax\100\bs4096_lr0p001_steps10755_l12p5_oldst.pth |
| fashion_mnist_st_100_softmax_lr0p001_oldst | 1 | 0.2h | models\fashion_mnist\st\softmax\100\bs4096_lr0p001_steps10755_l15p0_oldst.pth |
| fashion_mnist_st_100_softmax_lr0p001_l110p0_oldst | 1 | 0.2h | models\fashion_mnist\st\softmax\100\bs4096_lr0p001_steps10755_l110p0_oldst.pth |
| fashion_mnist_st_200_softmax_l12p5_oldst | 1 | 0.4h | models\fashion_mnist\st\softmax\200\bs4096_lr5e-05_steps18105_l12p5_oldst.pth |
| fashion_mnist_st_200_softmax_oldst | 1 | 0.4h | models\fashion_mnist\st\softmax\200\bs4096_lr5e-05_steps18105_l15p0_oldst.pth |
| fashion_mnist_st_200_softmax_l110p0_oldst | 1 | 0.4h | models\fashion_mnist\st\softmax\200\bs4096_lr5e-05_steps18105_l110p0_oldst.pth |
| fashion_mnist_st_200_softmax_lr0p001_l12p5_oldst | 1 | 0.4h | models\fashion_mnist\st\softmax\200\bs4096_lr0p001_steps18105_l12p5_oldst.pth |
| fashion_mnist_st_200_softmax_lr0p001_l110p0_oldst | 1 | 0.4h | models\fashion_mnist\st\softmax\200\bs4096_lr0p001_steps18105_l110p0_oldst.pth |
| fashion_mnist_st_200_softmax_lr0p001_oldst | 1 | 0.4h | models\fashion_mnist\st\softmax\200\bs4096_lr0p001_steps18105_l15p0_oldst.pth |
| fashion_mnist_st_400_softmax_l12p5_oldst | 1 | 0.6h | models\fashion_mnist\st\softmax\400\bs4096_lr5e-05_steps30450_l12p5_oldst.pth |
| fashion_mnist_st_400_softmax_l110p0_oldst | 1 | 0.6h | models\fashion_mnist\st\softmax\400\bs4096_lr5e-05_steps30450_l110p0_oldst.pth |
| fashion_mnist_st_400_softmax_oldst | 1 | 0.6h | models\fashion_mnist\st\softmax\400\bs4096_lr5e-05_steps30450_l15p0_oldst.pth |
| fashion_mnist_st_400_softmax_lr0p001_l12p5_oldst | 1 | 0.6h | models\fashion_mnist\st\softmax\400\bs4096_lr0p001_steps30450_l12p5_oldst.pth |
| fashion_mnist_st_400_softmax_lr0p001_oldst | 1 | 0.6h | models\fashion_mnist\st\softmax\400\bs4096_lr0p001_steps30450_l15p0_oldst.pth |
| fashion_mnist_st_400_softmax_lr0p001_l110p0_oldst | 1 | 0.6h | models\fashion_mnist\st\softmax\400\bs4096_lr0p001_steps30450_l110p0_oldst.pth |
| fashion_mnist_st_800_softmax_l12p5_oldst | 1 | 1.1h | models\fashion_mnist\st\softmax\800\bs4096_lr5e-05_steps51210_l12p5_oldst.pth |
| fashion_mnist_st_800_softmax_oldst | 1 | 1.1h | models\fashion_mnist\st\softmax\800\bs4096_lr5e-05_steps51210_l15p0_oldst.pth |
| fashion_mnist_st_800_softmax_l110p0_oldst | 1 | 1.1h | models\fashion_mnist\st\softmax\800\bs4096_lr5e-05_steps51210_l110p0_oldst.pth |
| fashion_mnist_st_800_softmax_lr0p001_l12p5_oldst | 1 | 1.1h | models\fashion_mnist\st\softmax\800\bs4096_lr0p001_steps51210_l12p5_oldst.pth |
| fashion_mnist_st_800_softmax_lr0p001_oldst | 1 | 1.1h | models\fashion_mnist\st\softmax\800\bs4096_lr0p001_steps51210_l15p0_oldst.pth |
| fashion_mnist_st_800_softmax_lr0p001_l110p0_oldst | 1 | 1.1h | models\fashion_mnist\st\softmax\800\bs4096_lr0p001_steps51210_l110p0_oldst.pth |
| fashion_mnist_st_1600_softmax_l110p0_oldst | 1 | 2.6h | models\fashion_mnist\st\softmax\1600\bs4096_lr5e-05_steps86145_l110p0_oldst.pth |
| fashion_mnist_st_1600_softmax_oldst | 1 | 2.6h | models\fashion_mnist\st\softmax\1600\bs4096_lr5e-05_steps86145_l15p0_oldst.pth |
| fashion_mnist_st_1600_softmax_l12p5_oldst | 1 | 2.6h | models\fashion_mnist\st\softmax\1600\bs4096_lr5e-05_steps86145_l12p5_oldst.pth |
| fashion_mnist_st_1600_softmax_lr0p001_l12p5_oldst | 1 | 2.5h | models\fashion_mnist\st\softmax\1600\bs4096_lr0p001_steps86145_l12p5_oldst.pth |
| fashion_mnist_st_1600_softmax_lr0p001_oldst | 1 | 2.5h | models\fashion_mnist\st\softmax\1600\bs4096_lr0p001_steps86145_l15p0_oldst.pth |
| fashion_mnist_st_1600_softmax_lr0p001_l110p0_oldst | 1 | 2.5h | models\fashion_mnist\st\softmax\1600\bs4096_lr0p001_steps86145_l110p0_oldst.pth |
| fashion_mnist_st_20_relu_softmax_l12p5_oldst | 1 | 0.1h | models\fashion_mnist\st\relu_softmax\20\bs4096_lr5e-05_steps6000_l12p5_oldst.pth |
| fashion_mnist_st_20_relu_softmax_oldst | 1 | 0.1h | models\fashion_mnist\st\relu_softmax\20\bs4096_lr5e-05_steps6000_l15p0_oldst.pth |
| fashion_mnist_st_20_relu_softmax_l110p0_oldst | 1 | 0.1h | models\fashion_mnist\st\relu_softmax\20\bs4096_lr5e-05_steps6000_l110p0_oldst.pth |
| fashion_mnist_st_20_relu_softmax_lr0p001_l12p5_oldst | 1 | 0.1h | models\fashion_mnist\st\relu_softmax\20\bs4096_lr0p001_steps6000_l12p5_oldst.pth |
| fashion_mnist_st_20_relu_softmax_lr0p001_oldst | 1 | 0.1h | models\fashion_mnist\st\relu_softmax\20\bs4096_lr0p001_steps6000_l15p0_oldst.pth |
| fashion_mnist_st_20_relu_softmax_lr0p001_l110p0_oldst | 1 | 0.1h | models\fashion_mnist\st\relu_softmax\20\bs4096_lr0p001_steps6000_l110p0_oldst.pth |
| fashion_mnist_st_50_relu_softmax_l12p5_oldst | 1 | 0.1h | models\fashion_mnist\st\relu_softmax\50\bs4096_lr5e-05_steps6390_l12p5_oldst.pth |
| fashion_mnist_st_50_relu_softmax_oldst | 1 | 0.1h | models\fashion_mnist\st\relu_softmax\50\bs4096_lr5e-05_steps6390_l15p0_oldst.pth |
| fashion_mnist_st_50_relu_softmax_l110p0_oldst | 1 | 0.1h | models\fashion_mnist\st\relu_softmax\50\bs4096_lr5e-05_steps6390_l110p0_oldst.pth |
| fashion_mnist_st_50_relu_softmax_lr0p001_l12p5_oldst | 1 | 0.1h | models\fashion_mnist\st\relu_softmax\50\bs4096_lr0p001_steps6390_l12p5_oldst.pth |
| fashion_mnist_st_50_relu_softmax_lr0p001_oldst | 1 | 0.1h | models\fashion_mnist\st\relu_softmax\50\bs4096_lr0p001_steps6390_l15p0_oldst.pth |
| fashion_mnist_st_50_relu_softmax_lr0p001_l110p0_oldst | 1 | 0.1h | models\fashion_mnist\st\relu_softmax\50\bs4096_lr0p001_steps6390_l110p0_oldst.pth |
| fashion_mnist_st_100_relu_softmax_l12p5_oldst | 1 | 0.2h | models\fashion_mnist\st\relu_softmax\100\bs4096_lr5e-05_steps10755_l12p5_oldst.pth |
| fashion_mnist_st_100_relu_softmax_oldst | 1 | 0.2h | models\fashion_mnist\st\relu_softmax\100\bs4096_lr5e-05_steps10755_l15p0_oldst.pth |
| fashion_mnist_st_100_relu_softmax_l110p0_oldst | 1 | 0.2h | models\fashion_mnist\st\relu_softmax\100\bs4096_lr5e-05_steps10755_l110p0_oldst.pth |
| fashion_mnist_st_100_relu_softmax_lr0p001_l12p5_oldst | 1 | 0.2h | models\fashion_mnist\st\relu_softmax\100\bs4096_lr0p001_steps10755_l12p5_oldst.pth |
| fashion_mnist_st_100_relu_softmax_lr0p001_oldst | 1 | 0.2h | models\fashion_mnist\st\relu_softmax\100\bs4096_lr0p001_steps10755_l15p0_oldst.pth |
| fashion_mnist_st_100_relu_softmax_lr0p001_l110p0_oldst | 1 | 0.2h | models\fashion_mnist\st\relu_softmax\100\bs4096_lr0p001_steps10755_l110p0_oldst.pth |
| fashion_mnist_st_200_relu_softmax_l12p5_oldst | 1 | 0.4h | models\fashion_mnist\st\relu_softmax\200\bs4096_lr5e-05_steps18105_l12p5_oldst.pth |
| fashion_mnist_st_200_relu_softmax_oldst | 1 | 0.4h | models\fashion_mnist\st\relu_softmax\200\bs4096_lr5e-05_steps18105_l15p0_oldst.pth |
| fashion_mnist_st_200_relu_softmax_l110p0_oldst | 1 | 0.4h | models\fashion_mnist\st\relu_softmax\200\bs4096_lr5e-05_steps18105_l110p0_oldst.pth |
| fashion_mnist_st_200_relu_softmax_lr0p001_l12p5_oldst | 1 | 0.4h | models\fashion_mnist\st\relu_softmax\200\bs4096_lr0p001_steps18105_l12p5_oldst.pth |
| fashion_mnist_st_200_relu_softmax_lr0p001_oldst | 1 | 0.4h | models\fashion_mnist\st\relu_softmax\200\bs4096_lr0p001_steps18105_l15p0_oldst.pth |
| fashion_mnist_st_200_relu_softmax_lr0p001_l110p0_oldst | 1 | 0.4h | models\fashion_mnist\st\relu_softmax\200\bs4096_lr0p001_steps18105_l110p0_oldst.pth |
| fashion_mnist_st_400_relu_softmax_l12p5_oldst | 1 | 0.6h | models\fashion_mnist\st\relu_softmax\400\bs4096_lr5e-05_steps30450_l12p5_oldst.pth |
| fashion_mnist_st_400_relu_softmax_oldst | 1 | 0.6h | models\fashion_mnist\st\relu_softmax\400\bs4096_lr5e-05_steps30450_l15p0_oldst.pth |
| fashion_mnist_st_400_relu_softmax_l110p0_oldst | 1 | 0.6h | models\fashion_mnist\st\relu_softmax\400\bs4096_lr5e-05_steps30450_l110p0_oldst.pth |
| fashion_mnist_st_400_relu_softmax_lr0p001_l12p5_oldst | 1 | 0.6h | models\fashion_mnist\st\relu_softmax\400\bs4096_lr0p001_steps30450_l12p5_oldst.pth |
| fashion_mnist_st_400_relu_softmax_lr0p001_oldst | 1 | 0.6h | models\fashion_mnist\st\relu_softmax\400\bs4096_lr0p001_steps30450_l15p0_oldst.pth |
| fashion_mnist_st_400_relu_softmax_lr0p001_l110p0_oldst | 1 | 0.6h | models\fashion_mnist\st\relu_softmax\400\bs4096_lr0p001_steps30450_l110p0_oldst.pth |
| fashion_mnist_st_800_relu_softmax_l12p5_oldst | 1 | 1.2h | models\fashion_mnist\st\relu_softmax\800\bs4096_lr5e-05_steps51210_l12p5_oldst.pth |
| fashion_mnist_st_800_relu_softmax_oldst | 1 | 1.2h | models\fashion_mnist\st\relu_softmax\800\bs4096_lr5e-05_steps51210_l15p0_oldst.pth |
| fashion_mnist_st_800_relu_softmax_l110p0_oldst | 1 | 1.2h | models\fashion_mnist\st\relu_softmax\800\bs4096_lr5e-05_steps51210_l110p0_oldst.pth |
| fashion_mnist_st_800_relu_softmax_lr0p001_l12p5_oldst | 1 | 1.2h | models\fashion_mnist\st\relu_softmax\800\bs4096_lr0p001_steps51210_l12p5_oldst.pth |
| fashion_mnist_st_800_relu_softmax_lr0p001_l110p0_oldst | 1 | 1.2h | models\fashion_mnist\st\relu_softmax\800\bs4096_lr0p001_steps51210_l110p0_oldst.pth |
| fashion_mnist_st_800_relu_softmax_lr0p001_oldst | 1 | 1.2h | models\fashion_mnist\st\relu_softmax\800\bs4096_lr0p001_steps51210_l15p0_oldst.pth |
| fashion_mnist_st_1600_relu_softmax_l110p0_oldst | 1 | 2.4h | models\fashion_mnist\st\relu_softmax\1600\bs4096_lr5e-05_steps86145_l110p0_oldst.pth |
| fashion_mnist_st_1600_relu_softmax_l12p5_oldst | 1 | 2.4h | models\fashion_mnist\st\relu_softmax\1600\bs4096_lr5e-05_steps86145_l12p5_oldst.pth |
| fashion_mnist_st_1600_relu_softmax_oldst | 1 | 2.4h | models\fashion_mnist\st\relu_softmax\1600\bs4096_lr5e-05_steps86145_l15p0_oldst.pth |
| fashion_mnist_st_1600_relu_softmax_lr0p001_oldst | 1 | 2.4h | models\fashion_mnist\st\relu_softmax\1600\bs4096_lr0p001_steps86145_l15p0_oldst.pth |
| fashion_mnist_st_1600_relu_softmax_lr0p001_l12p5_oldst | 1 | 2.4h | models\fashion_mnist\st\relu_softmax\1600\bs4096_lr0p001_steps86145_l12p5_oldst.pth |
| fashion_mnist_st_1600_relu_softmax_lr0p001_l110p0_oldst | 1 | 2.4h | models\fashion_mnist\st\relu_softmax\1600\bs4096_lr0p001_steps86145_l110p0_oldst.pth |
| fashion_mnist_st_20_tanh_scale_shift_l12p5_oldst | 1 | 0.1h | models\fashion_mnist\st\tanh_scale_shift\20\bs4096_lr5e-05_steps6000_l12p5_oldst.pth |
| fashion_mnist_st_20_tanh_scale_shift_oldst | 1 | 0.1h | models\fashion_mnist\st\tanh_scale_shift\20\bs4096_lr5e-05_steps6000_l15p0_oldst.pth |
| fashion_mnist_st_20_tanh_scale_shift_l110p0_oldst | 1 | 0.1h | models\fashion_mnist\st\tanh_scale_shift\20\bs4096_lr5e-05_steps6000_l110p0_oldst.pth |
| fashion_mnist_st_20_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 0.1h | models\fashion_mnist\st\tanh_scale_shift\20\bs4096_lr0p001_steps6000_l12p5_oldst.pth |
| fashion_mnist_st_20_tanh_scale_shift_lr0p001_oldst | 1 | 0.1h | models\fashion_mnist\st\tanh_scale_shift\20\bs4096_lr0p001_steps6000_l15p0_oldst.pth |
| fashion_mnist_st_20_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 0.1h | models\fashion_mnist\st\tanh_scale_shift\20\bs4096_lr0p001_steps6000_l110p0_oldst.pth |
| fashion_mnist_st_50_tanh_scale_shift_l12p5_oldst | 1 | 0.1h | models\fashion_mnist\st\tanh_scale_shift\50\bs4096_lr5e-05_steps6390_l12p5_oldst.pth |
| fashion_mnist_st_50_tanh_scale_shift_oldst | 1 | 0.1h | models\fashion_mnist\st\tanh_scale_shift\50\bs4096_lr5e-05_steps6390_l15p0_oldst.pth |
| fashion_mnist_st_50_tanh_scale_shift_l110p0_oldst | 1 | 0.1h | models\fashion_mnist\st\tanh_scale_shift\50\bs4096_lr5e-05_steps6390_l110p0_oldst.pth |
| fashion_mnist_st_50_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 0.1h | models\fashion_mnist\st\tanh_scale_shift\50\bs4096_lr0p001_steps6390_l12p5_oldst.pth |
| fashion_mnist_st_50_tanh_scale_shift_lr0p001_oldst | 1 | 0.1h | models\fashion_mnist\st\tanh_scale_shift\50\bs4096_lr0p001_steps6390_l15p0_oldst.pth |
| fashion_mnist_st_50_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 0.1h | models\fashion_mnist\st\tanh_scale_shift\50\bs4096_lr0p001_steps6390_l110p0_oldst.pth |
| fashion_mnist_st_100_tanh_scale_shift_l12p5_oldst | 1 | 0.2h | models\fashion_mnist\st\tanh_scale_shift\100\bs4096_lr5e-05_steps10755_l12p5_oldst.pth |
| fashion_mnist_st_100_tanh_scale_shift_l110p0_oldst | 1 | 0.2h | models\fashion_mnist\st\tanh_scale_shift\100\bs4096_lr5e-05_steps10755_l110p0_oldst.pth |
| fashion_mnist_st_100_tanh_scale_shift_oldst | 1 | 0.2h | models\fashion_mnist\st\tanh_scale_shift\100\bs4096_lr5e-05_steps10755_l15p0_oldst.pth |
| fashion_mnist_st_100_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 0.2h | models\fashion_mnist\st\tanh_scale_shift\100\bs4096_lr0p001_steps10755_l12p5_oldst.pth |
| fashion_mnist_st_100_tanh_scale_shift_lr0p001_oldst | 1 | 0.2h | models\fashion_mnist\st\tanh_scale_shift\100\bs4096_lr0p001_steps10755_l15p0_oldst.pth |
| fashion_mnist_st_100_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 0.2h | models\fashion_mnist\st\tanh_scale_shift\100\bs4096_lr0p001_steps10755_l110p0_oldst.pth |
| fashion_mnist_st_200_tanh_scale_shift_l12p5_oldst | 1 | 0.4h | models\fashion_mnist\st\tanh_scale_shift\200\bs4096_lr5e-05_steps18105_l12p5_oldst.pth |
| fashion_mnist_st_200_tanh_scale_shift_l110p0_oldst | 1 | 0.4h | models\fashion_mnist\st\tanh_scale_shift\200\bs4096_lr5e-05_steps18105_l110p0_oldst.pth |
| fashion_mnist_st_200_tanh_scale_shift_oldst | 1 | 0.4h | models\fashion_mnist\st\tanh_scale_shift\200\bs4096_lr5e-05_steps18105_l15p0_oldst.pth |
| fashion_mnist_st_200_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 0.4h | models\fashion_mnist\st\tanh_scale_shift\200\bs4096_lr0p001_steps18105_l12p5_oldst.pth |
| fashion_mnist_st_200_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 0.4h | models\fashion_mnist\st\tanh_scale_shift\200\bs4096_lr0p001_steps18105_l110p0_oldst.pth |
| fashion_mnist_st_200_tanh_scale_shift_lr0p001_oldst | 1 | 0.4h | models\fashion_mnist\st\tanh_scale_shift\200\bs4096_lr0p001_steps18105_l15p0_oldst.pth |
| fashion_mnist_st_400_tanh_scale_shift_l12p5_oldst | 1 | 0.6h | models\fashion_mnist\st\tanh_scale_shift\400\bs4096_lr5e-05_steps30450_l12p5_oldst.pth |
| fashion_mnist_st_400_tanh_scale_shift_oldst | 1 | 0.6h | models\fashion_mnist\st\tanh_scale_shift\400\bs4096_lr5e-05_steps30450_l15p0_oldst.pth |
| fashion_mnist_st_400_tanh_scale_shift_l110p0_oldst | 1 | 0.6h | models\fashion_mnist\st\tanh_scale_shift\400\bs4096_lr5e-05_steps30450_l110p0_oldst.pth |
| fashion_mnist_st_400_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 0.6h | models\fashion_mnist\st\tanh_scale_shift\400\bs4096_lr0p001_steps30450_l12p5_oldst.pth |
| fashion_mnist_st_400_tanh_scale_shift_lr0p001_oldst | 1 | 0.6h | models\fashion_mnist\st\tanh_scale_shift\400\bs4096_lr0p001_steps30450_l15p0_oldst.pth |
| fashion_mnist_st_400_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 0.6h | models\fashion_mnist\st\tanh_scale_shift\400\bs4096_lr0p001_steps30450_l110p0_oldst.pth |
| fashion_mnist_st_800_tanh_scale_shift_l12p5_oldst | 1 | 1.1h | models\fashion_mnist\st\tanh_scale_shift\800\bs4096_lr5e-05_steps51210_l12p5_oldst.pth |
| fashion_mnist_st_800_tanh_scale_shift_l110p0_oldst | 1 | 1.1h | models\fashion_mnist\st\tanh_scale_shift\800\bs4096_lr5e-05_steps51210_l110p0_oldst.pth |
| fashion_mnist_st_800_tanh_scale_shift_oldst | 1 | 1.1h | models\fashion_mnist\st\tanh_scale_shift\800\bs4096_lr5e-05_steps51210_l15p0_oldst.pth |
| fashion_mnist_st_800_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 1.1h | models\fashion_mnist\st\tanh_scale_shift\800\bs4096_lr0p001_steps51210_l12p5_oldst.pth |
| fashion_mnist_st_800_tanh_scale_shift_lr0p001_oldst | 1 | 1.1h | models\fashion_mnist\st\tanh_scale_shift\800\bs4096_lr0p001_steps51210_l15p0_oldst.pth |
| fashion_mnist_st_800_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 1.1h | models\fashion_mnist\st\tanh_scale_shift\800\bs4096_lr0p001_steps51210_l110p0_oldst.pth |
| fashion_mnist_st_1600_tanh_scale_shift_l110p0_oldst | 1 | 2.3h | models\fashion_mnist\st\tanh_scale_shift\1600\bs4096_lr5e-05_steps86145_l110p0_oldst.pth |
| fashion_mnist_st_1600_tanh_scale_shift_oldst | 1 | 2.3h | models\fashion_mnist\st\tanh_scale_shift\1600\bs4096_lr5e-05_steps86145_l15p0_oldst.pth |
| fashion_mnist_st_1600_tanh_scale_shift_l12p5_oldst | 1 | 2.3h | models\fashion_mnist\st\tanh_scale_shift\1600\bs4096_lr5e-05_steps86145_l12p5_oldst.pth |
| fashion_mnist_st_1600_tanh_scale_shift_lr0p001_l12p5_oldst | 1 | 2.3h | models\fashion_mnist\st\tanh_scale_shift\1600\bs4096_lr0p001_steps86145_l12p5_oldst.pth |
| fashion_mnist_st_1600_tanh_scale_shift_lr0p001_l110p0_oldst | 1 | 2.3h | models\fashion_mnist\st\tanh_scale_shift\1600\bs4096_lr0p001_steps86145_l110p0_oldst.pth |
| fashion_mnist_st_1600_tanh_scale_shift_lr0p001_oldst | 1 | 2.3h | models\fashion_mnist\st\tanh_scale_shift\1600\bs4096_lr0p001_steps86145_l15p0_oldst.pth |

