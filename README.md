# XAI Image Classifier - Optimized Production Version

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)  
[![Gradio](https://img.shields.io/badge/Gradio-Interface-4A90E2?style=flat&logo=gradio&logoColor=white)](https://gradio.app)  
[![License](https://img.shields.io/badge/License-MIT-4CAF50?style=flat&logo=github&logoColor=white)](LICENSE)

> **See *exactly* what the AI sees.**  
An **interactive, real-time explainable image classifier** powered by **ResNet50 + Grad-CAM** — built for **transparency**, **trust**, and **production use**.

---

## Features

| Feature | Description |
|-------|-------------|
| **1000-class ImageNet Prediction** | Top-5 predictions with confidence scores |
| **Grad-CAM Heatmaps** | Visual explanation of model attention |
| **Interactive Web UI** | Powered by Gradio — drag, drop, analyze |
| **GPU Acceleration** | CUDA support for fast inference |
| **Production-Ready** | Deploy locally or on Hugging Face, AWS, GCP, etc. |

---

## How It Works

1. **Upload** an image (JPG, PNG)  
2. Image is **preprocessed** (resize, normalize per ImageNet)  
3. **ResNet50** predicts top 5 classes  
4. **Grad-CAM** computes attention heatmap for the top prediction  
5. **Overlay visualization** shows *where* the model focused  
6. **Interpret**, debug, and build trust in AI decisions

> Perfect for auditing, education, and high-stakes applications.

---

## Tech Stack

| Component | Technology |
|---------|------------|
| **Model** | `torchvision.models.resnet50(pretrained=True)` |
| **Explainability** | `captum.attr.GradCAM` |
| **Interface** | `Gradio` (live demo in seconds) |
| **Backend** | `PyTorch`, `PIL`, `NumPy` |
| **Deployment** | Hugging Face Spaces, Docker, Cloud VMs |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/0AnshuAditya0/xai.git
cd xai
pip install -r requirements.txt
```

### 2. Run Locally

```bash
python app.py
```

Open the Gradio link in your browser → Upload image → Click "Analyze"

### Requirements

* Python >= 3.9
* PyTorch >= 2.0
* Gradio >= 3.0
* CUDA-compatible GPU (optional, but recommended)

### GPU Setup Tip:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Deployment Options

| Platform | How |
|----------|-----|
| Hugging Face Spaces | Push to HF → Auto-deploy with app.py |
| AWS / GCP / Azure | Deploy via Docker or FastAPI |
| Local / On-Prem | Run python app.py behind reverse proxy |

### Links

* Live Demo: https://huggingface.co/spaces/your-username/xai (replace with your link)
* GitHub: github.com/0AnshuAditya0/xai

### Real-World Use Cases

| Domain | Application |
|--------|-------------|
| Healthcare | Validate AI focus on tumors in X-rays |
| Autonomous Driving | Confirm detection of pedestrians, signs |
| Security | Audit surveillance AI for fairness |
| Education | Teach students how CNNs "see" |
| Ethics & Bias | Detect shortcut learning (e.g. background reliance) |

### Author

Anshu Aditya
AI Engineer | Building Transparent & Responsible AI
<img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin" alt="LinkedIn">
<img src="https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github" alt="GitHub">

MIT License — Free for research, learning, and production use