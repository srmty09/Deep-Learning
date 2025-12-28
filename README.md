# Deep Learning Repository

A comprehensive collection of deep learning implementations, research papers, and educational resources.

## Implemented Models and Architectures

### Computer Vision
- **AlexNet**: ImageNet Classification with Deep Convolutional Neural Networks
- **ResNet**: Deep Residual Learning for Image Recognition (ResNet-18)
- **LeNet**: Gradient-Based Learning Applied to Document Recognition
- **GoogleNet**: Going Deeper with Convolutions (Inception architecture)
- **MobileNet**: Efficient Convolutional Neural Networks for Mobile Vision Applications
- **Vision Transformer (ViT)**: An Image is Worth 16x16 Words
- **YOLO v1**: You Only Look Once - Real-Time Object Detection

### Natural Language Processing and Transformers
- **GPT-2**: Language Models are Unsupervised Multitask Learners
- **Transformer**: Attention Is All You Need (Neural Machine Translation)
- **Educational GPT**: Decoder-only transformer with causal attention
- **BERT**: Bidirectional Encoder Representations from Transformers
- **ALBERT**: A Lite BERT for Self-supervised Learning of Language Representations
- **DeBERTa**: Decoding-enhanced BERT with Disentangled Attention

### Recurrent Neural Networks
- **LSTM**: Long Short-Term Memory
- **Bidirectional RNN**: Bidirectional Recurrent Neural Networks
- **Deep RNN**: Multi-layer recurrent architectures
- **Encoder-Decoder**: Sequence to Sequence Learning with Neural Networks
- **GRU**: Learning Phrase Representations using RNN Encoder-Decoder

### Tokenization
- **BPE (Python)**: Neural Machine Translation of Rare Words with Subword Units
- **BPE (C++)**: High-performance C++ implementation of Byte Pair Encoding

### Advanced Architectures
- **Self-Attention with Relative Position Representations**
- **Autoencoder**: Dimensionality reduction and representation learning
- **LLaMA**: Large Language Model Meta AI architecture

### Optimization
- **Custom Optimizers**: SGD, Adam, AdamW, RMSprop implementations

## Projects and Applications

### Computer Vision Projects
- **CNN Training Pipeline**: Complete training system for CIFAR-10 and MNIST
- **Model Comparison**: Performance analysis across different CNN architectures
- **Inference System**: Real-time inference with visualization

### NLP Projects
- **BERT Fine-tuning**: Sentiment analysis with pre-trained BERT
- **Semantic Search**: Document similarity using embeddings
- **Neural Machine Translation**: English-Italian translation system

### Research and Analysis
- **Autoencoder Visualization**: Latent space analysis and reconstruction
- **Attention Visualization**: Understanding transformer attention patterns
- **Performance Metrics**: Comprehensive evaluation across models

## Repository Structure

```
Deep-Learning/
├── Computer Vision/
│   ├── CNNs/                    # CNN implementations and training
│   ├── ViT/                     # Vision Transformer
│   └── YOLO/                    # Object detection
├── Transformers/
│   ├── BERT/                    # BERT implementation
│   ├── GPT-2/                   # GPT-2 from scratch
│   ├── ALBERT/                  # ALBERT model
│   ├── DeBERTa/                 # DeBERTa implementation
│   ├── llama/                   # LLaMA architecture
│   └── lets build gpt/          # Educational GPT
├── RNNs/                        # Recurrent neural networks
├── NMT/                         # Neural machine translation
├── Tokenizers/                  # BPE implementations
├── Autoencoders/               # Autoencoder implementations
├── Optimizers/                 # Custom optimizer implementations
├── Projects/                   # End-to-end applications
├── Papers/                     # Research papers collection
└── Paper Notes/                # Paper summaries and analysis
```

## Research Papers Implemented

### Computer Vision Papers
- ImageNet Classification with Deep Convolutional Neural Networks (AlexNet, 2012)
- Deep Residual Learning for Image Recognition (ResNet, 2016)
- Going Deeper with Convolutions (GoogleNet, 2015)
- MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications (2017)
- An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT, 2021)
- You Only Look Once: Unified, Real-Time Object Detection (YOLO, 2016)
- Gradient-Based Learning Applied to Document Recognition (LeNet, 1998)

### NLP and Transformer Papers
- Attention Is All You Need (Transformer, 2017)
- Language Models are Unsupervised Multitask Learners (GPT-2, 2019)
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2019)
- ALBERT: A Lite BERT for Self-supervised Learning of Language Representations (2020)
- DeBERTa: Decoding-enhanced BERT with Disentangled Attention (2021)
- Neural Machine Translation of Rare Words with Subword Units (BPE, 2016)

### RNN Papers
- Long Short-Term Memory (LSTM, 1997)
- Sequence to Sequence Learning with Neural Networks (2014)
- Bidirectional Recurrent Neural Networks (1997)
- Learning Phrase Representations using RNN Encoder-Decoder (GRU, 2014)

### Additional Papers
- Self-Attention with Relative Position Representations (2018)
- Efficient Estimation of Word Representations in Vector Space (Word2Vec, 2013)

## Key Features

- **From Scratch Implementations**: All models implemented from first principles
- **Educational Focus**: Clear, well-documented code for learning
- **Paper Integration**: Code implementations paired with research papers
- **Multiple Domains**: Computer vision, NLP, and sequence modeling
- **Performance Analysis**: Training metrics and model comparisons
- **Production Ready**: Both research and optimized implementations

## Getting Started

1. **Computer Vision**: Start with `Computer Vision/CNNs/` for CNN implementations
2. **Transformers**: Explore `Transformers/lets build gpt/` for educational transformer implementation
3. **Projects**: Check `Projects/` for complete applications
4. **Papers**: Reference `Papers/` for theoretical background

## Hardware Requirements

- GPU recommended for training (CUDA support)
- Minimum 8GB RAM for most models
- Storage requirements vary by dataset

Made by **smruti**

