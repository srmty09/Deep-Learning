# Computer Vision

This directory contains implementations of various computer vision models and architectures from foundational papers.

## Implemented Models

### CNNs (Convolutional Neural Networks)
- **AlexNet**: ImageNet Classification with Deep Convolutional Neural Networks (Krizhevsky et al., 2012)
- **ResNet-18**: Deep Residual Learning for Image Recognition (He et al., 2016)
- **LeNet**: Gradient-Based Learning Applied to Document Recognition (LeCun et al., 1998)
- **GoogleNet**: Going Deeper with Convolutions (Szegedy et al., 2015)
- **MobileNet**: Efficient Convolutional Neural Networks for Mobile Vision Applications (Howard et al., 2017)

### Vision Transformer (ViT)
- **Implementation**: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2021)
- **Architecture**: Patch-based transformer for image classification
- **Features**: Multi-head self-attention, position embeddings, classification token

### YOLO (You Only Look Once)
- **YOLO v1**: You Only Look Once: Unified, Real-Time Object Detection (Redmon et al., 2016)
- **Architecture**: Single-stage object detection with grid-based predictions
- **Features**: Real-time inference, end-to-end training, unified detection framework

## Directory Structure

```
Computer Vision/
├── CNNs/
│   ├── Models/
│   │   ├── alex_net.py          # AlexNet implementation
│   │   ├── resnet.py            # ResNet-18 with residual blocks
│   │   ├── le_net.py            # LeNet for MNIST
│   │   ├── google_net.py        # GoogleNet with Inception modules
│   │   └── mobile_net.py        # MobileNet with depthwise separable convolutions
│   ├── main.py                  # Training pipeline
│   ├── training.py              # Training utilities
│   ├── inference.py             # Model inference and testing
│   ├── plotting.py              # Visualization tools
│   └── assets/                  # Training results and metrics
├── ViT/
│   ├── model.py                 # Vision Transformer implementation
│   └── test.ipynb               # ViT experimentation notebook
└── YOLO/
    └── yolo-v1/
        ├── model.py             # YOLO v1 architecture
        ├── dataset.py           # Dataset handling
        ├── loss.py              # YOLO loss function
        ├── train.py             # Training script
        └── utils.py             # Utility functions
```

## Training Results and Metrics

### CNN Performance (CIFAR-10)
- **AlexNet**: Achieved competitive accuracy with data augmentation
- **ResNet-18**: Best performance with residual connections
- **LeNet**: Optimized for MNIST digit recognition
- **GoogleNet**: Efficient parameter usage with Inception modules
- **MobileNet**: Reduced parameters while maintaining accuracy

### Vision Transformer
- **Patch Size**: 32x32 for 224x224 images
- **Architecture**: 12 layers, 12 attention heads, 768 hidden dimensions
- **Performance**: Competitive with CNNs on image classification

### YOLO v1
- **Grid Size**: 7x7 grid cells
- **Bounding Boxes**: 2 boxes per grid cell
- **Classes**: 20 classes (Pascal VOC format)
- **Speed**: Real-time object detection capability

## Key Innovations Implemented

### AlexNet (2012)
- Deep CNN architecture (8 layers)
- ReLU activation functions
- Dropout regularization
- Local Response Normalization
- Data augmentation techniques

### ResNet (2016)
- Residual connections (skip connections)
- Batch normalization
- Deep network training (18+ layers)
- Identity mapping for gradient flow

### GoogleNet (2015)
- Inception modules with parallel convolutions
- 1x1 convolutions for dimensionality reduction
- Auxiliary classifiers for training
- Efficient parameter usage

### MobileNet (2017)
- Depthwise separable convolutions
- Reduced computational cost
- Mobile-optimized architecture
- Width and resolution multipliers

### Vision Transformer (2021)
- Patch-based image processing
- Multi-head self-attention for images
- Position embeddings for spatial information
- Transformer architecture adapted for vision

### YOLO v1 (2016)
- Single-stage object detection
- Grid-based bounding box prediction
- End-to-end training
- Real-time inference capability

## Usage Examples

### CNN Training
```bash
cd CNNs/
python main.py  # Train all CNN models on CIFAR-10
python inference.py  # Run inference and generate predictions
```

### Vision Transformer
```python
from ViT.model import ViT, ViTConfig
config = ViTConfig()
model = ViT(config)
```

### YOLO Object Detection
```bash
cd YOLO/yolo-v1/
python train.py  # Train YOLO v1 model
```

## Datasets Supported
- **CIFAR-10**: 10-class image classification
- **MNIST**: Handwritten digit recognition
- **Pascal VOC**: Object detection (YOLO)
- **Custom datasets**: Extensible data loading pipeline

## Performance Metrics
All models include comprehensive evaluation:
- Training and validation accuracy curves
- Loss progression during training
- Inference time measurements
- Model parameter counts
- Memory usage analysis

## Hardware Requirements
- GPU recommended for training (CUDA support)
- Minimum 4GB GPU memory for most models
- 8GB+ GPU memory recommended for larger models
- CPU training supported but significantly slower

## Papers and References

### Implemented Papers
1. **AlexNet**: Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks.
2. **ResNet**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
3. **LeNet**: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition.
4. **GoogleNet**: Szegedy, C., et al. (2015). Going deeper with convolutions.
5. **MobileNet**: Howard, A. G., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications.
6. **Vision Transformer**: Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale.
7. **YOLO**: Redmon, J., et al. (2016). You only look once: Unified, real-time object detection.
