---
title: XAI Image Classifier
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
tags:
  - computer-vision
  - image-classification
  - explainable-ai
  - grad-cam
  - resnet
  - pytorch
  - interpretability
---

# 🔬 XAI Image Classifier: ResNet-152 with Grad-CAM

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44-orange?logo=gradio)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Production-grade explainable image classification** powered by ResNet-152 architecture with gradient-based visual attribution via Grad-CAM.

## 🎯 Overview

This space provides **transparent AI decision-making** for image classification tasks. Built on ResNet-152 (82.3% ImageNet Top-1 accuracy), it integrates Captum's LayerGradCam to generate pixel-level attribution maps, revealing which spatial regions drive class-specific predictions.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🧠 ResNet-152 Architecture** | 60M parameters, 82.3% ImageNet accuracy |
| **🔥 Grad-CAM Visualization** | Gradient-weighted class activation mapping |
| **⚡ GPU-Optimized Inference** | FP16 mixed-precision (~4-5ms latency on A100) |
| **📊 Multi-View Analysis** | Original + Heatmap + Overlay + Contours |
| **🎨 1000 ImageNet Classes** | Comprehensive object recognition |

## 🚀 How to Use

1. **Upload an image** (JPG, PNG, WebP supported)
2. Click **"🚀 Analyze"** to run inference
3. View **Top-10 predictions** with confidence scores
4. Examine **Grad-CAM heatmaps** showing model attention
5. Compare **multiple colormap visualizations**

## 🔬 Technical Architecture
```python
Model: ResNet-152 (torchvision.models.resnet152)
Weights: IMAGENET1K_V2 (pretrained)
XAI Method: Layer Grad-CAM (Captum)
Target Layer: layer4[-1] (final conv block)
Input Size: 224×224 RGB
Precision: FP16 (GPU) / FP32 (CPU)
```

### Performance Metrics

| Hardware | Inference Time | Memory Usage |
|----------|---------------|--------------|
| NVIDIA A100 | ~3-4ms | 1.2GB |
| NVIDIA T4 | ~8-10ms | 1.2GB |
| CPU (16 cores) | ~200ms | 2.5GB |

## 📊 Model Accuracy

- **Top-1 Accuracy:** 82.3% (ImageNet validation set)
- **Top-5 Accuracy:** 96.1%
- **Parameter Count:** 60.2M
- **FLOPs:** 11.6B

## 🛠️ Optimizations Applied

- **FP16 Mixed Precision:** 2x inference speedup on GPU
- **cuDNN Benchmark:** Auto-tuned convolution algorithms
- **TF32 Operations:** 8x faster matmuls on Ampere GPUs
- **Gradient Checkpointing:** Memory-efficient Grad-CAM computation

## 🎨 Visualization Outputs

1. **Original Image** - Input as-is
2. **Grad-CAM Heatmap** - Pure activation visualization
3. **Overlay** - Heatmap superimposed on original
4. **Multi-Colormap Comparison** - Jet, Hot, Viridis with contours

## 📖 Use Cases

| Domain | Application |
|--------|-------------|
| **Medical Imaging** | Validate diagnostic AI attention regions |
| **Autonomous Systems** | Debug object detection focus |
| **Security & Surveillance** | Audit algorithmic decision-making |
| **Research** | Study CNN feature representations |
| **Education** | Teach explainable AI concepts |

## 🔒 Privacy & Ethics

- ✅ **No data retention** - Images processed in-memory only
- ✅ **Zero telemetry** - No usage tracking
- ✅ **Open source** - Full code transparency
- ✅ **Bias auditing** - Visual inspection of model biases

## 📚 References

### Model Architecture
- He, K., et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.

### Explainability Method
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV.

### Framework
- PyTorch Team. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.

## 🔗 Links

- **GitHub Repository:** [0AnshuAditya0/xai](https://github.com/0AnshuAditya0/xai)
- **Documentation:** [Full Technical Docs](https://github.com/0AnshuAditya0/xai/wiki)
- **Paper (Grad-CAM):** [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- **Paper (ResNet):** [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

## ⚙️ Technical Requirements
```bash
# Core Dependencies
torch>=2.0.0
torchvision>=0.15.0
gradio>=4.44.0
captum>=0.6.0
Pillow>=9.0.0
numpy>=1.23.0
matplotlib>=3.5.0
```

## 🐛 Known Limitations

- **Memory:** Requires ~1.2GB GPU memory (FP16 mode)
- **Latency:** CPU inference slower (~200ms vs ~5ms GPU)
- **Classes:** Limited to 1000 ImageNet categories
- **Input Format:** RGB images only (grayscale not supported)

## 🔮 Roadmap

- [ ] Add support for custom model fine-tuning
- [ ] Implement batch processing API
- [ ] Integrate additional XAI methods (SHAP, Integrated Gradients)
- [ ] Add uncertainty quantification
- [ ] Support for video frame analysis

## 📄 License

MIT License - Free for research, education, and commercial use.

## 👨‍💻 Author

**Anshu Aditya**  
AI Engineer | Explainable AI Researcher

[![GitHub](https://img.shields.io/badge/GitHub-0AnshuAditya0-181717?logo=github)](https://github.com/0AnshuAditya0)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://linkedin.com/in/your-profile)

---

<div align="center">

**Built with ❤️ for transparent and accountable AI**

*Making deep learning interpretable, one image at a time*

⭐ Star this space if you find it useful!

</div>