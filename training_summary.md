# Training Results Summary

Date: 2025-04-16 11:28:46

## Overview

Total configurations: 27
Total runs: 27
Runs per configuration: 1
Successful: 0
Skipped: 0
Failed: 27

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

For GPT Neo, a special structure is used:
```
models/
  |-- gptneo/
      |-- layer{layer_number}/         # e.g., layer0, layer6, layer12
          |-- sae/
          |   |-- [activation_function]/
          |   |   |-- [feature_dimension]/
          |-- st/
              |-- [attention_function]/
                  |-- [feature_dimension]/
```

## Failed Models

- gptneo_layer8_st_4000_relu_attention_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer8_features.npz
- gptneo_layer9_st_4000_relu_attention_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer9_features.npz
- gptneo_layer10_st_4000_relu_attention_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer10_features.npz
- gptneo_layer11_st_4000_relu_attention_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer11_features.npz
- gptneo_layer12_st_4000_relu_attention_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer12_features.npz
- gptneo_layer13_st_4000_relu_attention_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer13_features.npz
- gptneo_layer14_st_4000_relu_attention_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer14_features.npz
- gptneo_layer15_st_4000_relu_attention_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer15_features.npz
- gptneo_layer16_st_4000_relu_attention_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer16_features.npz
- gptneo_layer8_st_4000_softmax_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer8_features.npz
- gptneo_layer9_st_4000_softmax_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer9_features.npz
- gptneo_layer10_st_4000_softmax_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer10_features.npz
- gptneo_layer11_st_4000_softmax_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer11_features.npz
- gptneo_layer12_st_4000_softmax_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer12_features.npz
- gptneo_layer13_st_4000_softmax_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer13_features.npz
- gptneo_layer14_st_4000_softmax_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer14_features.npz
- gptneo_layer15_st_4000_softmax_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer15_features.npz
- gptneo_layer16_st_4000_softmax_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer16_features.npz
- gptneo_layer8_st_4000_tanh_scale_shift_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer8_features.npz
- gptneo_layer9_st_4000_tanh_scale_shift_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer9_features.npz
- gptneo_layer10_st_4000_tanh_scale_shift_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer10_features.npz
- gptneo_layer11_st_4000_tanh_scale_shift_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer11_features.npz
- gptneo_layer12_st_4000_tanh_scale_shift_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer12_features.npz
- gptneo_layer13_st_4000_tanh_scale_shift_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer13_features.npz
- gptneo_layer14_st_4000_tanh_scale_shift_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer14_features.npz
- gptneo_layer15_st_4000_tanh_scale_shift_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer15_features.npz
- gptneo_layer16_st_4000_tanh_scale_shift_bs128_accum32 (return code: 1) - Features file not found: gptneo_features\randomized\layer16_features.npz
