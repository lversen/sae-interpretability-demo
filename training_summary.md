# Training Results Summary

Date: 2025-04-04 04:43:16

## Overview

Total configurations: 252
Total runs: 252
Runs per configuration: 1
Successful: 251
Skipped: 0
Failed: 1

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
| fashion_mnist_sae_20_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr0p001_steps4994_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr0p001_steps4994_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr0p001_steps4994_l110p0.pth |
| fashion_mnist_sae_20 | 1 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr5e-05_steps4994_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr5e-05_steps4994_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr5e-05_steps4994_l15p0.pth |
| fashion_mnist_sae_20_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr0p001_steps4994_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr0p001_steps4994_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr0p001_steps4994_l12p5.pth |
| fashion_mnist_sae_20_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr5e-05_steps4994_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr5e-05_steps4994_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr5e-05_steps4994_l110p0.pth |
| fashion_mnist_sae_20_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr5e-05_steps4994_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr5e-05_steps4994_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr5e-05_steps4994_l12p5.pth |
| fashion_mnist_sae_20_lr0p001 | 1 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr0p001_steps4994_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr0p001_steps4994_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\20\bs4096_lr0p001_steps4994_l15p0.pth |
| fashion_mnist_sae_50 | 1 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr5e-05_steps5324_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr5e-05_steps5324_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr5e-05_steps5324_l15p0.pth |
| fashion_mnist_sae_50_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr5e-05_steps5324_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr5e-05_steps5324_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr5e-05_steps5324_l12p5.pth |
| fashion_mnist_sae_50_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr0p001_steps5324_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr0p001_steps5324_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr0p001_steps5324_l110p0.pth |
| fashion_mnist_sae_50_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr0p001_steps5324_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr0p001_steps5324_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr0p001_steps5324_l12p5.pth |
| fashion_mnist_sae_50_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr5e-05_steps5324_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr5e-05_steps5324_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr5e-05_steps5324_l110p0.pth |
| fashion_mnist_sae_50_lr0p001 | 1 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr0p001_steps5324_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr0p001_steps5324_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\50\bs4096_lr0p001_steps5324_l15p0.pth |
| fashion_mnist_sae_100 | 1 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr5e-05_steps8969_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr5e-05_steps8969_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr5e-05_steps8969_l15p0.pth |
| fashion_mnist_sae_100_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr5e-05_steps8969_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr5e-05_steps8969_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr5e-05_steps8969_l12p5.pth |
| fashion_mnist_sae_100_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr5e-05_steps8969_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr5e-05_steps8969_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr5e-05_steps8969_l110p0.pth |
| fashion_mnist_sae_100_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr0p001_steps8969_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr0p001_steps8969_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr0p001_steps8969_l110p0.pth |
| fashion_mnist_sae_100_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr0p001_steps8969_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr0p001_steps8969_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr0p001_steps8969_l12p5.pth |
| fashion_mnist_sae_100_lr0p001 | 1 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr0p001_steps8969_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr0p001_steps8969_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\100\bs4096_lr0p001_steps8969_l15p0.pth |
| fashion_mnist_sae_200 | 1 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr5e-05_steps15089_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr5e-05_steps15089_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr5e-05_steps15089_l15p0.pth |
| fashion_mnist_sae_200_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr5e-05_steps15089_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr5e-05_steps15089_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr5e-05_steps15089_l12p5.pth |
| fashion_mnist_sae_200_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr5e-05_steps15089_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr5e-05_steps15089_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr5e-05_steps15089_l110p0.pth |
| fashion_mnist_sae_200_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr0p001_steps15089_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr0p001_steps15089_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr0p001_steps15089_l110p0.pth |
| fashion_mnist_sae_200_lr0p001 | 1 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr0p001_steps15089_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr0p001_steps15089_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr0p001_steps15089_l15p0.pth |
| fashion_mnist_sae_200_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr0p001_steps15089_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr0p001_steps15089_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\200\bs4096_lr0p001_steps15089_l12p5.pth |
| fashion_mnist_sae_400 | 1 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr5e-05_steps25379_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr5e-05_steps25379_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr5e-05_steps25379_l15p0.pth |
| fashion_mnist_sae_400_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr5e-05_steps25379_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr5e-05_steps25379_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr5e-05_steps25379_l12p5.pth |
| fashion_mnist_sae_400_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr5e-05_steps25379_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr5e-05_steps25379_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr5e-05_steps25379_l110p0.pth |
| fashion_mnist_sae_400_lr0p001 | 1 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr0p001_steps25379_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr0p001_steps25379_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr0p001_steps25379_l15p0.pth |
| fashion_mnist_sae_400_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr0p001_steps25379_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr0p001_steps25379_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr0p001_steps25379_l12p5.pth |
| fashion_mnist_sae_400_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr0p001_steps25379_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr0p001_steps25379_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\400\bs4096_lr0p001_steps25379_l110p0.pth |
| fashion_mnist_sae_800 | 1 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr5e-05_steps42674_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr5e-05_steps42674_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr5e-05_steps42674_l15p0.pth |
| fashion_mnist_sae_800_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr5e-05_steps42674_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr5e-05_steps42674_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr5e-05_steps42674_l12p5.pth |
| fashion_mnist_sae_800_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr5e-05_steps42674_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr5e-05_steps42674_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr5e-05_steps42674_l110p0.pth |
| fashion_mnist_sae_800_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr0p001_steps42674_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr0p001_steps42674_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr0p001_steps42674_l110p0.pth |
| fashion_mnist_sae_800_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr0p001_steps42674_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr0p001_steps42674_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr0p001_steps42674_l12p5.pth |
| fashion_mnist_sae_800_lr0p001 | 1 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr0p001_steps42674_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr0p001_steps42674_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\800\bs4096_lr0p001_steps42674_l15p0.pth |
| fashion_mnist_sae_1600 | 1 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr5e-05_steps71789_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr5e-05_steps71789_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr5e-05_steps71789_l15p0.pth |
| fashion_mnist_sae_1600_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr5e-05_steps71789_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr5e-05_steps71789_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr5e-05_steps71789_l12p5.pth |
| fashion_mnist_sae_1600_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr5e-05_steps71789_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr5e-05_steps71789_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr5e-05_steps71789_l110p0.pth |
| fashion_mnist_sae_1600_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr0p001_steps71789_l12p5.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr0p001_steps71789_l12p5.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr0p001_steps71789_l12p5.pth |
| fashion_mnist_sae_1600_lr0p001 | 1 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr0p001_steps71789_l15p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr0p001_steps71789_l15p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr0p001_steps71789_l15p0.pth |
| fashion_mnist_sae_1600_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr0p001_steps71789_l110p0.pth |
| | 2 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr0p001_steps71789_l110p0.pth |
| | 3 | 0.0h | models\fashion_mnist\sae\relu\1600\bs4096_lr0p001_steps71789_l110p0.pth |
| fashion_mnist_st_20_softmax | 1 | 0.0h | models\fashion_mnist\st\softmax\20\bs4096_lr5e-05_steps6000_l15p0.pth |
| fashion_mnist_st_20_softmax_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\20\bs4096_lr5e-05_steps6000_l12p5.pth |
| fashion_mnist_st_20_softmax_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\softmax\20\bs4096_lr0p001_steps6000_l15p0.pth |
| fashion_mnist_st_20_softmax_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\20\bs4096_lr5e-05_steps6000_l110p0.pth |
| fashion_mnist_st_20_softmax_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\20\bs4096_lr0p001_steps6000_l110p0.pth |
| fashion_mnist_st_20_softmax_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\20\bs4096_lr0p001_steps6000_l12p5.pth |
| fashion_mnist_st_50_softmax | 1 | 0.0h | models\fashion_mnist\st\softmax\50\bs4096_lr5e-05_steps6390_l15p0.pth |
| fashion_mnist_st_50_softmax_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\50\bs4096_lr5e-05_steps6390_l12p5.pth |
| fashion_mnist_st_50_softmax_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\50\bs4096_lr0p001_steps6390_l12p5.pth |
| fashion_mnist_st_50_softmax_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\50\bs4096_lr5e-05_steps6390_l110p0.pth |
| fashion_mnist_st_50_softmax_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\50\bs4096_lr0p001_steps6390_l110p0.pth |
| fashion_mnist_st_50_softmax_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\softmax\50\bs4096_lr0p001_steps6390_l15p0.pth |
| fashion_mnist_st_100_softmax | 1 | 0.0h | models\fashion_mnist\st\softmax\100\bs4096_lr5e-05_steps10755_l15p0.pth |
| fashion_mnist_st_100_softmax_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\100\bs4096_lr5e-05_steps10755_l12p5.pth |
| fashion_mnist_st_100_softmax_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\100\bs4096_lr5e-05_steps10755_l110p0.pth |
| fashion_mnist_st_100_softmax_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\100\bs4096_lr0p001_steps10755_l12p5.pth |
| fashion_mnist_st_100_softmax_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\softmax\100\bs4096_lr0p001_steps10755_l15p0.pth |
| fashion_mnist_st_100_softmax_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\100\bs4096_lr0p001_steps10755_l110p0.pth |
| fashion_mnist_st_200_softmax | 1 | 0.0h | models\fashion_mnist\st\softmax\200\bs4096_lr5e-05_steps18105_l15p0.pth |
| fashion_mnist_st_200_softmax_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\200\bs4096_lr5e-05_steps18105_l12p5.pth |
| fashion_mnist_st_200_softmax_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\200\bs4096_lr0p001_steps18105_l12p5.pth |
| fashion_mnist_st_200_softmax_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\200\bs4096_lr5e-05_steps18105_l110p0.pth |
| fashion_mnist_st_200_softmax_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\softmax\200\bs4096_lr0p001_steps18105_l15p0.pth |
| fashion_mnist_st_200_softmax_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\200\bs4096_lr0p001_steps18105_l110p0.pth |
| fashion_mnist_st_400_softmax_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\400\bs4096_lr5e-05_steps30450_l12p5.pth |
| fashion_mnist_st_400_softmax | 1 | 0.0h | models\fashion_mnist\st\softmax\400\bs4096_lr5e-05_steps30450_l15p0.pth |
| fashion_mnist_st_400_softmax_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\400\bs4096_lr5e-05_steps30450_l110p0.pth |
| fashion_mnist_st_400_softmax_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\softmax\400\bs4096_lr0p001_steps30450_l15p0.pth |
| fashion_mnist_st_400_softmax_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\400\bs4096_lr0p001_steps30450_l12p5.pth |
| fashion_mnist_st_400_softmax_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\400\bs4096_lr0p001_steps30450_l110p0.pth |
| fashion_mnist_st_800_softmax | 1 | 0.0h | models\fashion_mnist\st\softmax\800\bs4096_lr5e-05_steps51210_l15p0.pth |
| fashion_mnist_st_800_softmax_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\800\bs4096_lr5e-05_steps51210_l12p5.pth |
| fashion_mnist_st_800_softmax_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\800\bs4096_lr0p001_steps51210_l12p5.pth |
| fashion_mnist_st_800_softmax_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\800\bs4096_lr5e-05_steps51210_l110p0.pth |
| fashion_mnist_st_800_softmax_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\800\bs4096_lr0p001_steps51210_l110p0.pth |
| fashion_mnist_st_800_softmax_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\softmax\800\bs4096_lr0p001_steps51210_l15p0.pth |
| fashion_mnist_st_1600_softmax | 1 | 0.0h | models\fashion_mnist\st\softmax\1600\bs4096_lr5e-05_steps86145_l15p0.pth |
| fashion_mnist_st_1600_softmax_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\1600\bs4096_lr5e-05_steps86145_l12p5.pth |
| fashion_mnist_st_20_relu_attention | 1 | 0.0h | models\fashion_mnist\st\relu_attention\20\bs4096_lr5e-05_steps6000_l15p0.pth |
| fashion_mnist_st_20_relu_attention_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\20\bs4096_lr5e-05_steps6000_l12p5.pth |
| fashion_mnist_st_1600_softmax_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\softmax\1600\bs4096_lr0p001_steps86145_l15p0.pth |
| fashion_mnist_st_1600_softmax_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\1600\bs4096_lr5e-05_steps86145_l110p0.pth |
| fashion_mnist_st_1600_softmax_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\softmax\1600\bs4096_lr0p001_steps86145_l110p0.pth |
| fashion_mnist_st_1600_softmax_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\softmax\1600\bs4096_lr0p001_steps86145_l12p5.pth |
| fashion_mnist_st_20_relu_attention_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\20\bs4096_lr5e-05_steps6000_l110p0.pth |
| fashion_mnist_st_20_relu_attention_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\20\bs4096_lr0p001_steps6000_l15p0.pth |
| fashion_mnist_st_20_relu_attention_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\20\bs4096_lr0p001_steps6000_l12p5.pth |
| fashion_mnist_st_20_relu_attention_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\20\bs4096_lr0p001_steps6000_l110p0.pth |
| fashion_mnist_st_50_relu_attention | 1 | 0.0h | models\fashion_mnist\st\relu_attention\50\bs4096_lr5e-05_steps6390_l15p0.pth |
| fashion_mnist_st_50_relu_attention_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\50\bs4096_lr5e-05_steps6390_l12p5.pth |
| fashion_mnist_st_50_relu_attention_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\50\bs4096_lr5e-05_steps6390_l110p0.pth |
| fashion_mnist_st_50_relu_attention_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\50\bs4096_lr0p001_steps6390_l15p0.pth |
| fashion_mnist_st_50_relu_attention_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\50\bs4096_lr0p001_steps6390_l12p5.pth |
| fashion_mnist_st_50_relu_attention_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\50\bs4096_lr0p001_steps6390_l110p0.pth |
| fashion_mnist_st_100_relu_attention | 1 | 0.0h | models\fashion_mnist\st\relu_attention\100\bs4096_lr5e-05_steps10755_l15p0.pth |
| fashion_mnist_st_100_relu_attention_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\100\bs4096_lr5e-05_steps10755_l12p5.pth |
| fashion_mnist_st_100_relu_attention_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\100\bs4096_lr5e-05_steps10755_l110p0.pth |
| fashion_mnist_st_100_relu_attention_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\100\bs4096_lr0p001_steps10755_l15p0.pth |
| fashion_mnist_st_100_relu_attention_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\100\bs4096_lr0p001_steps10755_l12p5.pth |
| fashion_mnist_st_100_relu_attention_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\100\bs4096_lr0p001_steps10755_l110p0.pth |
| fashion_mnist_st_200_relu_attention | 1 | 0.0h | models\fashion_mnist\st\relu_attention\200\bs4096_lr5e-05_steps18105_l15p0.pth |
| fashion_mnist_st_200_relu_attention_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\200\bs4096_lr5e-05_steps18105_l12p5.pth |
| fashion_mnist_st_200_relu_attention_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\200\bs4096_lr5e-05_steps18105_l110p0.pth |
| fashion_mnist_st_200_relu_attention_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\200\bs4096_lr0p001_steps18105_l15p0.pth |
| fashion_mnist_st_200_relu_attention_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\200\bs4096_lr0p001_steps18105_l12p5.pth |
| fashion_mnist_st_200_relu_attention_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\200\bs4096_lr0p001_steps18105_l110p0.pth |
| fashion_mnist_st_400_relu_attention | 1 | 0.0h | models\fashion_mnist\st\relu_attention\400\bs4096_lr5e-05_steps30450_l15p0.pth |
| fashion_mnist_st_400_relu_attention_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\400\bs4096_lr5e-05_steps30450_l12p5.pth |
| fashion_mnist_st_400_relu_attention_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\400\bs4096_lr5e-05_steps30450_l110p0.pth |
| fashion_mnist_st_400_relu_attention_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\400\bs4096_lr0p001_steps30450_l15p0.pth |
| fashion_mnist_st_400_relu_attention_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\400\bs4096_lr0p001_steps30450_l110p0.pth |
| fashion_mnist_st_400_relu_attention_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\400\bs4096_lr0p001_steps30450_l12p5.pth |
| fashion_mnist_st_800_relu_attention_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\800\bs4096_lr5e-05_steps51210_l12p5.pth |
| fashion_mnist_st_800_relu_attention | 1 | 0.0h | models\fashion_mnist\st\relu_attention\800\bs4096_lr5e-05_steps51210_l15p0.pth |
| fashion_mnist_st_800_relu_attention_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\800\bs4096_lr5e-05_steps51210_l110p0.pth |
| fashion_mnist_st_800_relu_attention_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\800\bs4096_lr0p001_steps51210_l15p0.pth |
| fashion_mnist_st_800_relu_attention_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\800\bs4096_lr0p001_steps51210_l12p5.pth |
| fashion_mnist_st_800_relu_attention_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\800\bs4096_lr0p001_steps51210_l110p0.pth |
| fashion_mnist_st_1600_relu_attention | 1 | 0.0h | models\fashion_mnist\st\relu_attention\1600\bs4096_lr5e-05_steps86145_l15p0.pth |
| fashion_mnist_st_1600_relu_attention_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\1600\bs4096_lr5e-05_steps86145_l12p5.pth |
| fashion_mnist_st_1600_relu_attention_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\1600\bs4096_lr5e-05_steps86145_l110p0.pth |
| fashion_mnist_st_1600_relu_attention_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\1600\bs4096_lr0p001_steps86145_l15p0.pth |
| fashion_mnist_st_1600_relu_attention_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\1600\bs4096_lr0p001_steps86145_l110p0.pth |
| fashion_mnist_st_1600_relu_attention_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\relu_attention\1600\bs4096_lr0p001_steps86145_l12p5.pth |
| fashion_mnist_st_20_tanh_scale_shift | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\20\bs4096_lr5e-05_steps6000_l15p0.pth |
| fashion_mnist_st_20_tanh_scale_shift_l12p5 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\20\bs4096_lr5e-05_steps6000_l12p5.pth |
| fashion_mnist_st_20_tanh_scale_shift_l110p0 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\20\bs4096_lr5e-05_steps6000_l110p0.pth |
| fashion_mnist_st_20_tanh_scale_shift_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\20\bs4096_lr0p001_steps6000_l15p0.pth |
| fashion_mnist_st_20_tanh_scale_shift_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\20\bs4096_lr0p001_steps6000_l110p0.pth |
| fashion_mnist_st_20_tanh_scale_shift_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\20\bs4096_lr0p001_steps6000_l12p5.pth |
| fashion_mnist_st_50_tanh_scale_shift | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\50\bs4096_lr5e-05_steps6390_l15p0.pth |
| fashion_mnist_st_50_tanh_scale_shift_l12p5 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\50\bs4096_lr5e-05_steps6390_l12p5.pth |
| fashion_mnist_st_50_tanh_scale_shift_l110p0 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\50\bs4096_lr5e-05_steps6390_l110p0.pth |
| fashion_mnist_st_50_tanh_scale_shift_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\50\bs4096_lr0p001_steps6390_l15p0.pth |
| fashion_mnist_st_50_tanh_scale_shift_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\50\bs4096_lr0p001_steps6390_l12p5.pth |
| fashion_mnist_st_50_tanh_scale_shift_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\50\bs4096_lr0p001_steps6390_l110p0.pth |
| fashion_mnist_st_100_tanh_scale_shift | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\100\bs4096_lr5e-05_steps10755_l15p0.pth |
| fashion_mnist_st_100_tanh_scale_shift_l12p5 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\100\bs4096_lr5e-05_steps10755_l12p5.pth |
| fashion_mnist_st_100_tanh_scale_shift_l110p0 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\100\bs4096_lr5e-05_steps10755_l110p0.pth |
| fashion_mnist_st_100_tanh_scale_shift_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\100\bs4096_lr0p001_steps10755_l15p0.pth |
| fashion_mnist_st_100_tanh_scale_shift_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\100\bs4096_lr0p001_steps10755_l110p0.pth |
| fashion_mnist_st_200_tanh_scale_shift | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\200\bs4096_lr5e-05_steps18105_l15p0.pth |
| fashion_mnist_st_100_tanh_scale_shift_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\100\bs4096_lr0p001_steps10755_l12p5.pth |
| fashion_mnist_st_200_tanh_scale_shift_l12p5 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\200\bs4096_lr5e-05_steps18105_l12p5.pth |
| fashion_mnist_st_200_tanh_scale_shift_l110p0 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\200\bs4096_lr5e-05_steps18105_l110p0.pth |
| fashion_mnist_st_200_tanh_scale_shift_lr0p001 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\200\bs4096_lr0p001_steps18105_l15p0.pth |
| fashion_mnist_st_200_tanh_scale_shift_lr0p001_l12p5 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\200\bs4096_lr0p001_steps18105_l12p5.pth |
| fashion_mnist_st_200_tanh_scale_shift_lr0p001_l110p0 | 1 | 0.0h | models\fashion_mnist\st\tanh_scale_shift\200\bs4096_lr0p001_steps18105_l110p0.pth |
| fashion_mnist_st_400_tanh_scale_shift_l12p5 | 1 | 0.8h | models\fashion_mnist\st\tanh_scale_shift\400\bs4096_lr5e-05_steps30450_l12p5.pth |
| fashion_mnist_st_400_tanh_scale_shift | 1 | 0.8h | models\fashion_mnist\st\tanh_scale_shift\400\bs4096_lr5e-05_steps30450_l15p0.pth |
| fashion_mnist_st_400_tanh_scale_shift_lr0p001 | 1 | 0.8h | models\fashion_mnist\st\tanh_scale_shift\400\bs4096_lr0p001_steps30450_l15p0.pth |
| fashion_mnist_st_400_tanh_scale_shift_lr0p001_l12p5 | 1 | 0.8h | models\fashion_mnist\st\tanh_scale_shift\400\bs4096_lr0p001_steps30450_l12p5.pth |
| fashion_mnist_st_400_tanh_scale_shift_lr0p001_l110p0 | 1 | 0.8h | models\fashion_mnist\st\tanh_scale_shift\400\bs4096_lr0p001_steps30450_l110p0.pth |
| fashion_mnist_st_400_tanh_scale_shift_l110p0 | 1 | 0.8h | models\fashion_mnist\st\tanh_scale_shift\400\bs4096_lr5e-05_steps30450_l110p0.pth |
| fashion_mnist_st_800_tanh_scale_shift | 1 | 1.6h | models\fashion_mnist\st\tanh_scale_shift\800\bs4096_lr5e-05_steps51210_l15p0.pth |
| fashion_mnist_st_800_tanh_scale_shift_lr0p001_l12p5 | 1 | 1.6h | models\fashion_mnist\st\tanh_scale_shift\800\bs4096_lr0p001_steps51210_l12p5.pth |
| fashion_mnist_st_800_tanh_scale_shift_l110p0 | 1 | 1.7h | models\fashion_mnist\st\tanh_scale_shift\800\bs4096_lr5e-05_steps51210_l110p0.pth |
| fashion_mnist_st_800_tanh_scale_shift_lr0p001 | 1 | 1.6h | models\fashion_mnist\st\tanh_scale_shift\800\bs4096_lr0p001_steps51210_l15p0.pth |
| fashion_mnist_st_800_tanh_scale_shift_l12p5 | 1 | 1.7h | models\fashion_mnist\st\tanh_scale_shift\800\bs4096_lr5e-05_steps51210_l12p5.pth |
| fashion_mnist_st_800_tanh_scale_shift_lr0p001_l110p0 | 1 | 1.6h | models\fashion_mnist\st\tanh_scale_shift\800\bs4096_lr0p001_steps51210_l110p0.pth |
| fashion_mnist_st_1600_tanh_scale_shift_lr0p001_l110p0 | 1 | 3.6h | models\fashion_mnist\st\tanh_scale_shift\1600\bs4096_lr0p001_steps86145_l110p0.pth |
| fashion_mnist_st_1600_tanh_scale_shift_lr0p001_l12p5 | 1 | 3.6h | models\fashion_mnist\st\tanh_scale_shift\1600\bs4096_lr0p001_steps86145_l12p5.pth |
| fashion_mnist_st_1600_tanh_scale_shift_lr0p001 | 1 | 3.6h | models\fashion_mnist\st\tanh_scale_shift\1600\bs4096_lr0p001_steps86145_l15p0.pth |
| fashion_mnist_st_1600_tanh_scale_shift_l12p5 | 1 | 3.6h | models\fashion_mnist\st\tanh_scale_shift\1600\bs4096_lr5e-05_steps86145_l12p5.pth |
| fashion_mnist_st_1600_tanh_scale_shift | 1 | 3.6h | models\fashion_mnist\st\tanh_scale_shift\1600\bs4096_lr5e-05_steps86145_l15p0.pth |

## Failed Models

- fashion_mnist_st_1600_tanh_scale_shift_l110p0 (return code: 1)
