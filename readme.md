# Simple RNN Implementation

This project implements a simple Recurrent Neural Network (RNN) using PyTorch to predict the next value in a sequence of numbers.

## Overview

The RNN model is trained on sequences of incremental values and learns to predict the next value in the sequence. The implementation includes:

- A `SimpleRNN` class that inherits from `nn.Module`
- Data generation function for training
- Training loop with Adam optimizer
- Evaluation on test sequence

## Model Architecture

The model consists of:

- Input size: $1$ (single feature)
- Hidden size: $16$ units
- Output size: $1$ (prediction)
- Number of RNN layers: $1$

## Requirements

```python
torch>=1.0.0
```

## Usage

Run the script directly:

```bash
python rnn.py
```

The script will:

1. Generate training data
2. Train the model for 100 epochs
3. Print loss every 10 epochs
4. Evaluate on a test sequence

## Model Parameters

- Learning rate: $0.01$
- Sequence length: $5$
- Number of training samples: $100$
- Loss function: MSE (Mean Squared Error)

## Data Generation

Training data is generated as sequences where:

- Each sequence starts with a random value between $0$ and $0.5$
- Subsequent values increment by $0.1$
- Sequence length is $5$ time steps
