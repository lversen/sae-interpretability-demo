# Training Results Summary

Date: 2025-03-24 10:08:01

## Overview

Total configurations: 2
Total runs: 6
Runs per configuration: 3
Successful: 6
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
| mnist_st_10_softmax_steps100 | 1 | 0.0h | models\mnist\st\softmax\10\bs4096_lr5e-05_steps100_3.pth |
| | 2 | 0.0h | models\mnist\st\softmax\10\bs4096_lr5e-05_steps100_4.pth |
| | 3 | 0.0h | models\mnist\st\softmax\10\bs4096_lr5e-05_steps100_5.pth |
| mnist_st_20_softmax_steps100 | 1 | 0.0h | models\mnist\st\softmax\20\bs4096_lr5e-05_steps100_3.pth |
| | 2 | 0.0h | models\mnist\st\softmax\20\bs4096_lr5e-05_steps100_4.pth |
| | 3 | 0.0h | models\mnist\st\softmax\20\bs4096_lr5e-05_steps100_5.pth |

