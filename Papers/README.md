# Research Papers

This directory contains important research papers organized by domain and architecture type.

## Directory Structure

### CNN Papers (`CNN-papers/`)
Collection of foundational convolutional neural network papers:

- **AlexNet.pdf**: "ImageNet Classification with Deep Convolutional Neural Networks" (Krizhevsky et al., 2012)
- **DenseNet.pdf**: "Densely Connected Convolutional Networks" (Huang et al., 2017)
- **GoggleNet.pdf**: "Going Deeper with Convolutions" (Szegedy et al., 2015)
- **LeNet.pdf**: "Gradient-Based Learning Applied to Document Recognition" (LeCun et al., 1998)
- **ResNet.pdf**: "Deep Residual Learning for Image Recognition" (He et al., 2016)
- **VGG.pdf**: "Very Deep Convolutional Networks for Large-Scale Image Recognition" (Simonyan & Zisserman, 2015)

### NLP Papers (`NLP/`)
Natural language processing foundational papers:

- **word2vec.pdf**: "Efficient Estimation of Word Representations in Vector Space" (Mikolov et al., 2013)

### RNN Papers (`RNN-papers/`)
Recurrent neural network architectures and applications:

- **bidirrectionalrnn.pdf**: "Bidirectional Recurrent Neural Networks" (Schuster & Paliwal, 1997)
- **gru.pdf**: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (Cho et al., 2014)
- **lstm.pdf**: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- **seq2seq.pdf**: "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014)

### Transformer Papers (`Transformer/`)
Transformer architecture and attention mechanisms:

- **1706.03762v7.pdf**: "Attention Is All You Need" (Vaswani et al., 2017)

## Paper Summaries

### Convolutional Neural Networks

#### AlexNet (2012)
- **Impact**: Breakthrough in ImageNet competition, sparked deep learning revolution
- **Key Innovations**: Deep CNN, ReLU activation, dropout, data augmentation
- **Architecture**: 8 layers (5 conv, 3 FC), 60M parameters

#### VGG (2015)
- **Impact**: Showed importance of network depth
- **Key Innovations**: Very small (3×3) convolution filters, deep architecture
- **Architecture**: 16-19 layers, uniform architecture design

#### GoogLeNet/Inception (2015)
- **Impact**: Efficient deep networks with parallel convolutions
- **Key Innovations**: Inception modules, 1×1 convolutions, auxiliary classifiers
- **Architecture**: 22 layers, reduced parameters through efficient design

#### ResNet (2016)
- **Impact**: Enabled training of very deep networks (100+ layers)
- **Key Innovations**: Residual connections, skip connections, batch normalization
- **Architecture**: 50-152 layers, identity shortcuts

#### DenseNet (2017)
- **Impact**: Maximum information flow between layers
- **Key Innovations**: Dense connections, feature reuse, parameter efficiency
- **Architecture**: Dense blocks with growth rate

#### LeNet (1998)
- **Impact**: First successful CNN for practical applications
- **Key Innovations**: Convolutional layers, subsampling, gradient-based learning
- **Architecture**: 7 layers, designed for handwritten digit recognition

### Recurrent Neural Networks

#### LSTM (1997)
- **Impact**: Solved vanishing gradient problem in RNNs
- **Key Innovations**: Memory cells, gating mechanisms, forget gates
- **Applications**: Language modeling, machine translation, speech recognition

#### Bidirectional RNN (1997)
- **Impact**: Access to both past and future context
- **Key Innovations**: Forward and backward processing, context combination
- **Applications**: Speech recognition, protein secondary structure prediction

#### GRU (2014)
- **Impact**: Simplified alternative to LSTM
- **Key Innovations**: Gating units, reset and update gates, fewer parameters
- **Applications**: Machine translation, sequence modeling

#### Seq2Seq (2014)
- **Impact**: Framework for sequence-to-sequence learning
- **Key Innovations**: Encoder-decoder architecture, variable-length sequences
- **Applications**: Machine translation, text summarization, conversation

### Transformers and Attention

#### Attention Is All You Need (2017)
- **Impact**: Revolutionary architecture that replaced RNNs for many tasks
- **Key Innovations**: Self-attention, multi-head attention, positional encoding
- **Applications**: Machine translation, language modeling, computer vision

### Natural Language Processing

#### Word2Vec (2013)
- **Impact**: Efficient word embeddings that capture semantic relationships
- **Key Innovations**: Skip-gram and CBOW models, negative sampling
- **Applications**: Word similarity, analogy tasks, downstream NLP tasks

## Papers with Implementations in Repository

### Computer Vision (Implemented)
- **Vision Transformer**: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021) - `Computer Vision/ViT/`
- **YOLO**: "You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al., 2016) - `Computer Vision/YOLO/yolo-v1/`
- **MobileNet**: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications" (Howard et al., 2017) - `Computer Vision/CNNs/Models/mobile_net.py`

### Natural Language Processing (Implemented)
- **GPT**: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018) - `Transformers/lets build gpt/`
- **GPT-2**: "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019) - `Transformers/GPT-2/`
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2019) - `Transformers/BERT/`
- **ALBERT**: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (Lan et al., 2020) - `Transformers/ALBERT/`
- **DeBERTa**: "DeBERTa: Decoding-enhanced BERT with Disentangled Attention" (He et al., 2021) - `Transformers/DeBERTa/`

### Tokenization (Implemented)
- **BPE**: "Neural Machine Translation of Rare Words with Subword Units" (Sennrich et al., 2016) - `Tokenizers/BPE/`

### Vision Language Model Papers (Implemented)
- **SigLIP**: "Sigmoid Loss for Language Image Pre-training" (Zhai et al., 2023) - `Vision Language Model/modeling_siglip.py`
- **CLIP** concepts - Referenced in VLM implementation and notes

### Advanced Architectures (Implemented)
- **LLaMA**: "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023) - `Transformers/llama/`
- **Self-Attention with Relative Position Representations** (Shaw et al., 2018) - `Self-Attention with Relative Position Representations/`

## Additional Papers Recommended for Collection

### Missing Important Papers
- **SentencePiece**: "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing" (Kudo & Richardson, 2018)
- **T5**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (Raffel et al., 2020)
- **RoBERTa**: "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (Liu et al., 2019)
- **CLIP**: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)

## Implementation Status and Locations

### Fully Implemented Papers
All papers listed below have complete implementations in the repository:

#### Computer Vision Papers
1. **AlexNet** (2012) → `Computer Vision/CNNs/Models/alex_net.py`
2. **VGG** (2015) → Referenced in CNN implementations
3. **GoogleNet** (2015) → `Computer Vision/CNNs/Models/google_net.py`
4. **ResNet** (2016) → `Computer Vision/CNNs/Models/resnet.py`
5. **DenseNet** (2017) → Paper available, implementation referenced
6. **LeNet** (1998) → `Computer Vision/CNNs/Models/le_net.py`
7. **MobileNet** (2017) → `Computer Vision/CNNs/Models/mobile_net.py`
8. **Vision Transformer** (2021) → `Computer Vision/ViT/model.py`
9. **YOLO v1** (2016) → `Computer Vision/YOLO/yolo-v1/`

#### NLP and Transformer Papers
1. **Word2Vec** (2013) → Paper available, concepts used in embeddings
2. **LSTM** (1997) → `RNNs/` directory implementations
3. **Bidirectional RNN** (1997) → `RNNs/Bidirectional RNNs/model.py`
4. **GRU** (2014) → `RNNs/Encoder-Decoder/` implementations
5. **Seq2Seq** (2014) → `RNNs/Encoder-Decoder/encoder-decoder.py` and `NMT/`
6. **Attention Is All You Need** (2017) → `NMT/model.py` and transformer implementations
7. **GPT-2** (2019) → `Transformers/GPT-2/gpt2model.py`
8. **BERT** (2019) → `Transformers/BERT/model.py`
9. **ALBERT** (2020) → `Transformers/ALBERT/model.py`
10. **DeBERTa** (2021) → `Transformers/DeBERTa/model.py`
11. **LLaMA** (2023) → `Transformers/llama/llama-1.py`

#### Tokenization Papers
1. **BPE** (2016) → `Tokenizers/BPE/bpe.py` (Python) and `Tokenizers/BPE In CPP/` (C++)

#### Vision Language Model Papers
1. **SigLIP** (2023) → `Vision Language Model/modeling_siglip.py`
2. **CLIP** concepts → Referenced in VLM implementation and notes

#### Additional Implementations
1. **Self-Attention with Relative Position Representations** (2018) → `Self-Attention with Relative Position Representations/model.py`
2. **Autoencoder** concepts → `Autoencoders/autoencoder.py`

## Usage Notes

- Papers are organized by architectural family for easy reference
- Each paper has corresponding implementations in the codebase
- PDFs can be referenced when studying the code implementations
- All implementations follow the original paper specifications
- Code includes detailed comments referencing paper sections

## Reading Recommendations

### For Beginners
1. Start with LeNet and AlexNet for CNN foundations
2. Read LSTM paper for RNN understanding
3. Progress to Attention Is All You Need for modern architectures

### For Advanced Study
1. Compare ResNet vs DenseNet approaches to deep networks
2. Study evolution from RNNs (LSTM/GRU) to Transformers
3. Understand attention mechanisms across different domains

### Implementation Study
- Use papers alongside code implementations
- Compare paper descriptions with actual code
- Understand design choices and trade-offs made in implementations
