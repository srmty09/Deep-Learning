# Vision Language Model (VLM)

This directory contains implementations of vision-language models that bridge computer vision and natural language processing.

## Implemented Models

### SigLIP (Sigmoid Loss for Language Image Pre-training)
- **File**: `modeling_siglip.py`
- **Paper**: "Sigmoid Loss for Language Image Pre-training" (Zhai et al., 2023)
- **Description**: Improved contrastive learning approach for vision-language understanding
- **Key Innovation**: Uses sigmoid loss instead of softmax for better computational efficiency

## Architecture Overview

### SigLIP Vision Transformer Components

#### SiglipVisionConfig
- Configuration class for SigLIP vision model
- **Parameters**:
  - `hidden_size`: 768 (embedding dimension)
  - `intermediate_size`: 3072 (MLP hidden dimension)
  - `num_hidden_layers`: 12 (transformer layers)
  - `num_attention_heads`: 12 (attention heads)
  - `image_size`: 224 (input image size)
  - `patch_size`: 16 (patch size for tokenization)

#### SiglipVisionEmbeddings
- Converts images to patch embeddings
- **Process**:
  1. Convolutional patch embedding (16x16 patches)
  2. Positional embeddings for spatial information
  3. Combines patch and position embeddings

#### SiglipAttention
- Multi-head self-attention mechanism
- **Features**:
  - Scaled dot-product attention
  - Configurable attention dropout
  - Multi-head parallel processing

#### SiglipMLP
- Feed-forward network with GELU activation
- **Architecture**:
  - Linear layer: hidden_size → intermediate_size
  - GELU activation (tanh approximation)
  - Linear layer: intermediate_size → hidden_size

#### SiglipEncoderLayer
- Single transformer encoder layer
- **Components**:
  - Layer normalization (pre-norm)
  - Multi-head self-attention
  - Residual connections
  - MLP block

#### SiglipEncoder
- Stack of transformer encoder layers
- **Configuration**: 12 layers by default
- **Processing**: Sequential application of encoder layers

#### SiglipVisionTransformer
- Complete vision transformer for SigLIP
- **Pipeline**:
  1. Patch embeddings
  2. Transformer encoder stack
  3. Post-layer normalization

#### SiglipVisionModel
- Top-level vision model wrapper
- **Input**: Pixel values (batch_size, channels, height, width)
- **Output**: Vision embeddings (batch_size, num_patches, embed_dim)

## Key Concepts and Innovations

### 1. Contrastive Learning
- **Objective**: Learn joint embeddings for images and text
- **Process**:
  - Image encoder produces image embeddings
  - Text encoder produces text embeddings
  - Contrastive loss maximizes similarity for matching pairs
  - Minimizes similarity for non-matching pairs

### 2. Sigmoid Loss vs Softmax Loss
- **Traditional Approach**: Softmax-based contrastive loss
- **SigLIP Innovation**: Sigmoid-based loss function
- **Advantages**:
  - Better computational efficiency
  - Improved numerical stability
  - Reduced memory requirements
  - Better scaling to large batch sizes

### 3. Vision Transformer Architecture
- **Patch-based Processing**: Images divided into 16x16 patches
- **Self-attention**: Global attention across all patches
- **Position Encoding**: Learnable positional embeddings
- **Hierarchical Features**: Multi-layer feature extraction

## Mathematical Formulation

### Contrastive Learning Loss
```
I_f = img_encoder(I)  # [n, d_i]
T_f = text_encoder(T) # [n, d_t]

I_e = l2_normalize(I_f @ W_i, axis=1)  # [n, d_e]
T_e = l2_normalize(T_f @ W_t, axis=1)  # [n, d_e]

logits = (I_e @ T_e.T) * exp(temperature)
```

### Sigmoid Loss Function
```
loss = (1/|β|) * Σᵢ Σⱼ log(1 / (1 + exp(-t * xᵢ · yⱼ + b)))
```

Where:
- `t` is temperature parameter
- `b` is bias term
- Binary classification for each image-text pair

## Directory Structure

```
Vision Language Model/
├── modeling_siglip.py          # Complete SigLIP implementation
└── Note/
    ├── note.md                 # Detailed concepts and theory
    ├── image.png               # Contrastive learning diagram
    └── image-1.png             # Softmax numerical stability
```

## Usage Example

```python
from modeling_siglip import SiglipVisionModel, SiglipVisionConfig

# Initialize configuration
config = SiglipVisionConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    image_size=224,
    patch_size=16
)

# Create vision model
vision_model = SiglipVisionModel(config)

# Forward pass
import torch
pixel_values = torch.randn(4, 3, 224, 224)  # Batch of images
vision_embeddings = vision_model(pixel_values)
# Output shape: (4, 196, 768) - 196 patches, 768-dim embeddings
```

## Key Features

### Vision Processing
- **Patch Tokenization**: Converts images to sequence of patches
- **Self-attention**: Global context across all image patches
- **Positional Encoding**: Maintains spatial relationships
- **Multi-scale Features**: Hierarchical feature extraction

### Efficiency Improvements
- **Sigmoid Loss**: More efficient than softmax for large vocabularies
- **Numerical Stability**: Improved handling of large logit values
- **Memory Efficiency**: Reduced memory requirements for training
- **Scalability**: Better performance with large batch sizes

### Architecture Benefits
- **Transfer Learning**: Pre-trained representations for downstream tasks
- **Multimodal Understanding**: Joint vision-language representations
- **Fine-tuning**: Adaptable to specific vision-language tasks
- **Interpretability**: Attention weights show visual focus areas

## Applications

### Vision-Language Tasks
- **Image-Text Retrieval**: Finding relevant images for text queries
- **Visual Question Answering**: Answering questions about images
- **Image Captioning**: Generating descriptions for images
- **Cross-modal Search**: Searching across vision and language modalities

### Downstream Applications
- **Zero-shot Classification**: Classifying images without task-specific training
- **Few-shot Learning**: Learning new concepts with minimal examples
- **Transfer Learning**: Adapting to new vision-language tasks
- **Multimodal Embeddings**: Joint representations for various applications

## Technical Notes

### Implementation Details
- **Pre-norm Architecture**: Layer normalization before attention and MLP
- **GELU Activation**: Gaussian Error Linear Unit with tanh approximation
- **Attention Dropout**: Configurable dropout for attention weights
- **Residual Connections**: Skip connections for gradient flow

### Training Considerations
- **Batch Size**: Large batches beneficial for contrastive learning
- **Temperature Scaling**: Important hyperparameter for contrastive loss
- **Learning Rate**: Careful scheduling for stable training
- **Data Augmentation**: Important for robust vision-language learning

## Paper Reference

"Sigmoid Loss for Language Image Pre-training"
- Authors: Zhai et al.
- Year: 2023
- Key Contribution: Sigmoid-based contrastive loss for improved efficiency

## Related Work

- **CLIP**: Learning Transferable Visual Models From Natural Language Supervision
- **ALIGN**: Scaling Up Visual and Vision-Language Representation Learning
- **Florence**: A New Foundation Model for Computer Vision

This implementation provides a foundation for understanding and building vision-language models with improved efficiency through sigmoid-based contrastive learning.

