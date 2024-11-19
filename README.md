# KAT Architecture

This repository implements a novel approach to image classification by integrating **Kolmogorov–Arnold Networks (KAN)** into the **Vision Transformer (ViT)** architecture. The implementation is inspired by the research paper *"Kolmogorov–Arnold Transformer (KAT)"*, which proposed replacing traditional MLP layers in transformers with more expressive and computationally efficient KAN layers.

## Introduction

### Kolmogorov–Arnold Transformer (KAT)
KAT replaces the Multi-Layer Perceptron (MLP) layers in transformers with **Kolmogorov–Arnold Networks (KAN)**. The KANs leverage learnable activation functions (e.g., rational functions) to enhance the model's expressiveness while maintaining computational efficiency. KAT introduces the **Group Rational KAN (GRKAN)** to address the limitations of original KAN designs, such as:
1. **Inefficient Base Functions**: KANs traditionally use B-splines, which are not GPU-friendly.
2. **Scalability Issues**: Unique activation functions for each input-output pair lead to excessive parameters and computations.
3. **Poor Initialization**: Default initialization strategies for KANs fail to preserve variance, resulting in unstable training.

### Our Implementation
This repository integrates GRKAN into the Vision Transformer (ViT) architecture to enhance its channel mixing capabilities. The key contributions are:
- Replacing MLP layers in ViT with GRKAN layers.
- Leveraging rational functions as the activation base for GRKAN.
- Preserving the original transformer attention mechanism while enhancing expressiveness.

## Features
- Fully functional **Vision Transformer (ViT)** with GRKAN for channel mixing.
- Implements **Rational Functions** for efficient GPU operations.
- Comprehensive training and testing pipelines on **CIFAR-10**.

## How It Works

### Vision Transformer with GRKAN
1. **Patch Embedding**: Converts input images into token embeddings.
2. **Positional Embedding**: Adds positional information to tokens.
3. **Self-Attention**: Standard transformer attention mechanism.
4. **GRKAN**: Replaces the MLP layer with Group Rational KAN for improved expressiveness and efficiency.

### GRKAN Design
- Divides input channels into groups and applies shared rational activation functions within each group.
- Rational functions are parameterized as:
  \[ \phi(x) = \frac{a_0 + a_1x + \ldots + a_mx^m}{1 + |b_0 + b_1x + \ldots + b_nx^n|} \]
- Efficient CUDA-friendly implementation for parallelization.

## Installation

### Requirements
- Python 3.7+
- PyTorch 1.10+
- torchvision
- numpy
- transformers

### Install Dependencies
```bash
pip install torch torchvision numpy transformers
