# Transformers

This directory contains implementations of transformer-based models and architectures from foundational research papers.

## Overview

Transformers are a neural network architecture that relies entirely on attention mechanisms to draw global dependencies between input and output. They have revolutionized natural language processing and are increasingly used in computer vision and other domains.

## Implemented Models

### BERT (Bidirectional Encoder Representations from Transformers)
- **Directory**: `BERT/`
- **Paper**: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2019)
- **Architecture**: Bidirectional encoder-only transformer
- **Features**: Masked language modeling, next sentence prediction, bidirectional context

### GPT-2 (Generative Pre-trained Transformer 2)
- **Directory**: `GPT-2/`
- **Paper**: Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- **Architecture**: Decoder-only transformer with causal attention
- **Features**: Autoregressive text generation, large-scale language modeling

### ALBERT (A Lite BERT)
- **Directory**: `ALBERT/`
- **Paper**: ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (Lan et al., 2020)
- **Architecture**: Parameter-efficient BERT variant
- **Features**: Parameter sharing, factorized embeddings, sentence order prediction

### DeBERTa (Decoding-enhanced BERT with Disentangled Attention)
- **Directory**: `DeBERTa/`
- **Paper**: DeBERTa: Decoding-enhanced BERT with Disentangled Attention (He et al., 2021)
- **Architecture**: Enhanced BERT with disentangled attention
- **Features**: Disentangled attention mechanism, enhanced mask decoder

### LLaMA (Large Language Model Meta AI)
- **Directory**: `llama/`
- **Paper**: LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
- **Architecture**: Decoder-only transformer with improvements
- **Features**: RMSNorm, SwiGLU activation, rotary positional embeddings

### Educational GPT ("lets build gpt")
- **Directory**: `lets build gpt/`
- **Description**: Educational implementation of GPT (Generative Pre-trained Transformer)
- **Features**: Decoder-only transformer, causal attention, character-level tokenization

## Directory Structure

```
Transformers/
├── BERT/
│   ├── model.py                 # BERT implementation
│   ├── dataset.py               # Dataset handling
│   ├── config.py                # Model configuration
│   ├── main.py                  # Training script
│   └── requirements.txt         # Dependencies
├── GPT-2/
│   ├── gpt2model.py            # GPT-2 implementation
│   ├── test.ipynb              # Testing notebook
│   └── Karypathy_Lecture/      # Educational materials
├── ALBERT/
│   ├── model.py                # ALBERT implementation
│   └── dataset.py              # Dataset utilities
├── DeBERTa/
│   └── model.py                # DeBERTa implementation
├── llama/
│   └── llama-1.py              # LLaMA implementation
└── lets build gpt/
    ├── basic_self_attention.ipynb  # Self-attention tutorial
    ├── bigram_model.ipynb         # Simple bigram baseline
    ├── model.py                   # GPT implementation
    └── input.txt                  # Training text
```

## GPT Implementation Details

### Architecture Components

#### Self-Attention
```python
class SelfAttention(nn.Module):
    - Single attention head implementation
    - Causal masking for autoregressive generation
    - Scaled dot-product attention
```

#### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    - Multiple parallel attention heads
    - Concatenation and projection of head outputs
    - Improved representation capacity
```

#### Feed-Forward Network
```python
class FeedForwardNetwork(nn.Module):
    - Position-wise fully connected layers
    - ReLU activation with dropout
    - 4x hidden dimension expansion
```

#### Transformer Block
```python
class Block(nn.Module):
    - Multi-head attention + residual connection
    - Feed-forward network + residual connection
    - Layer normalization (pre-norm architecture)
```

### Model Configuration

```python
# Hyperparameters
context_length = 8      # Maximum sequence length
d_model = 384          # Embedding dimension
h = 6                  # Number of attention heads
n_layers = 6           # Number of transformer blocks
n_vocab = 65           # Vocabulary size
dropout = 0.2          # Dropout rate
```

## Key Features

### Causal Self-Attention
- **Masking**: Lower triangular mask prevents future information leakage
- **Autoregressive**: Model generates one token at a time
- **Parallel Training**: All positions computed simultaneously during training

### Position-wise Processing
- **No Recurrence**: Unlike RNNs, processes all positions in parallel
- **Position Encoding**: Uses learned positional embeddings
- **Permutation Invariant**: Attention is inherently position-agnostic

### Residual Connections
- **Skip Connections**: Help with gradient flow in deep networks
- **Layer Normalization**: Stabilizes training
- **Pre-norm Architecture**: LayerNorm before attention and FFN

## Educational Notebooks

### `basic_self_attention.ipynb`
- Step-by-step implementation of self-attention
- Visualization of attention patterns
- Understanding of attention mechanics

### `bigram_model.ipynb`
- Simple baseline model for comparison
- Character-level bigram language model
- Demonstrates the power of transformers vs simple models

## Usage Example

```python
from model import Model

# Initialize model
model = Model()

# Generate text
input_ids = torch.randint(0, n_vocab, (batch_size, context_length))
logits = model(input_ids)  # Shape: (batch_size, context_length, vocab_size)

# For generation (autoregressive)
generated = model.generate(start_tokens, max_length=100)
```

## Training Process

1. **Character-Level Tokenization**: Simple character-to-integer mapping
2. **Causal Language Modeling**: Predict next character given previous characters
3. **Cross-Entropy Loss**: Standard loss for language modeling
4. **Teacher Forcing**: Use ground truth during training

## Key Innovations of Transformers

### Attention Mechanism
- **Global Dependencies**: Can attend to any position in sequence
- **Parallel Computation**: All positions processed simultaneously
- **Interpretability**: Attention weights show what model focuses on

### Scalability
- **Parameter Scaling**: Performance improves with model size
- **Data Scaling**: Benefits from large datasets
- **Compute Scaling**: Efficient use of modern hardware (GPUs/TPUs)

### Transfer Learning
- **Pre-training**: Learn general language understanding
- **Fine-tuning**: Adapt to specific tasks
- **Few-shot Learning**: Perform tasks with minimal examples

## Advantages Over RNNs

1. **Parallelization**: No sequential dependency in computation
2. **Long-range Dependencies**: Direct connections between distant positions
3. **Training Speed**: Faster training due to parallelization
4. **Performance**: Better results on most NLP tasks

## Applications

### Natural Language Processing
- **Language Modeling**: GPT series models
- **Machine Translation**: Encoder-decoder transformers
- **Text Classification**: BERT-style models
- **Question Answering**: Reading comprehension tasks

### Computer Vision
- **Vision Transformer (ViT)**: Image classification
- **DETR**: Object detection
- **Image Generation**: DALL-E style models

### Other Domains
- **Protein Folding**: AlphaFold uses transformer components
- **Code Generation**: GitHub Copilot, CodeT5
- **Music Generation**: MuseNet, Jukebox

## Implementation Notes

- **Attention Scaling**: Divide by sqrt(head_dimension) for stability
- **Causal Masking**: Essential for autoregressive generation
- **Dropout**: Applied to attention weights and FFN outputs
- **Weight Initialization**: Proper initialization crucial for training
- **Gradient Clipping**: Helps with training stability

## Implemented Papers and Architectures

### Core Transformer Papers
1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Original transformer paper introducing self-attention mechanism
   - Implemented in NMT directory as encoder-decoder architecture

2. **"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"** (Devlin et al., 2019)
   - Bidirectional encoder-only transformer
   - Masked language modeling and next sentence prediction
   - Implementation includes pre-training and fine-tuning capabilities

3. **"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019)
   - GPT-2 paper demonstrating scaling benefits
   - Complete implementation with causal attention and text generation

4. **"ALBERT: A Lite BERT for Self-supervised Learning of Language Representations"** (Lan et al., 2020)
   - Parameter-efficient BERT variant
   - Parameter sharing across layers and factorized embeddings

5. **"DeBERTa: Decoding-enhanced BERT with Disentangled Attention"** (He et al., 2021)
   - Enhanced BERT with disentangled attention mechanism
   - Improved performance on downstream tasks

6. **"LLaMA: Open and Efficient Foundation Language Models"** (Touvron et al., 2023)
   - Efficient large language model architecture
   - RMSNorm, SwiGLU activation, rotary positional embeddings

### Key Architectural Innovations Implemented

#### BERT Innovations
- **Bidirectional Context**: Unlike GPT, processes text bidirectionally
- **Masked Language Modeling**: Predicts masked tokens using full context
- **Next Sentence Prediction**: Learns sentence relationships
- **WordPiece Tokenization**: Subword tokenization for better vocabulary coverage

#### GPT-2 Innovations
- **Causal Self-Attention**: Autoregressive generation with masked attention
- **Layer Normalization**: Pre-norm architecture for training stability
- **Byte Pair Encoding**: Subword tokenization handling any text
- **Zero-shot Task Performance**: Performs tasks without fine-tuning

#### ALBERT Improvements
- **Parameter Sharing**: Shares parameters across transformer layers
- **Factorized Embeddings**: Separates vocabulary size from hidden size
- **Sentence Order Prediction**: Replaces next sentence prediction
- **Reduced Memory**: Significantly fewer parameters than BERT

#### DeBERTa Enhancements
- **Disentangled Attention**: Separates content and position representations
- **Enhanced Mask Decoder**: Improved handling of masked positions
- **Relative Position Encoding**: Better position understanding
- **Virtual Adversarial Training**: Improved robustness

#### LLaMA Optimizations
- **RMSNorm**: More efficient normalization than LayerNorm
- **SwiGLU Activation**: Improved activation function
- **Rotary Position Embeddings**: Better position encoding
- **Efficient Architecture**: Optimized for inference and training
