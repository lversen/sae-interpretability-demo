# Training Results Summary

Date: 2025-04-16 23:23:22

## Overview

Total configurations: 9
Total runs: 9
Runs per configuration: 1
Successful: 0
Skipped: 0
Failed: 9

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

## Failed Models

- gptneo_gpt_neo_1p3B_layer9_st_8000_softmax_bs128_accum32 (return code: 2)
- gptneo_gpt_neo_1p3B_layer10_st_8000_softmax_bs128_accum32 (return code: 2)
- gptneo_gpt_neo_1p3B_layer11_st_8000_softmax_bs128_accum32 (return code: 2)
- gptneo_gpt_neo_1p3B_layer8_st_8000_softmax_bs128_accum32 (return code: 2)
- gptneo_gpt_neo_1p3B_layer14_st_8000_softmax_bs128_accum32 (return code: 2)
- gptneo_gpt_neo_1p3B_layer15_st_8000_softmax_bs128_accum32 (return code: 2)
- gptneo_gpt_neo_1p3B_layer13_st_8000_softmax_bs128_accum32 (return code: 2)
- gptneo_gpt_neo_1p3B_layer12_st_8000_softmax_bs128_accum32 (return code: 2)
- gptneo_gpt_neo_1p3B_layer16_st_8000_softmax_bs128_accum32 (return code: 2)
