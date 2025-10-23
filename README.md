# XAI Image Classifier

An explainable image classification web app using ResNet18 fine-tuned on CIFAR-10, with Grad-CAM visualizations for interpretability.

## Features

- **Image Classification**: Classifies images into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
- **Explainability**: Uses Grad-CAM to show which parts of the image influenced the prediction.
- **Interactive UI**: Built with Gradio for easy web-based interaction.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/xai-image-classifier.git
   cd xai-image-classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the model file `xai_resnet18.pth` and place it in the `model/` directory.

## Usage

Run the app:
```bash
python app.py
```

Open the provided URL in your browser, upload an image, and click "Analyze Image" to get predictions and explanations.

## Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, for faster inference)
- Dependencies listed in `requirements.txt`

## Model Details

- **Architecture**: ResNet18 with modified fully-connected layer.
- **Training Data**: CIFAR-10 dataset (60,000 images).
- **Pre-trained Weights**: None (trained from scratch).

## License

MIT License
