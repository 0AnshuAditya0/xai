import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from captum.attr import LayerGradCam
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load model
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    
    checkpoint = torch.load('model/xai_resnet18.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    return model

model = load_model()
target_layer = model.layer4[1].conv2
gradcam = LayerGradCam(model, target_layer)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Prediction function
def predict_and_explain(image):
    if image is None:
        return "Please upload an image", None
    
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_class = probabilities.argmax(1).item()
        confidence = probabilities[0][pred_class].item()
    
    # Generate Grad-CAM
    attributions = gradcam.attribute(img_tensor, target=pred_class)
    attr_np = attributions.squeeze().cpu().detach().numpy()
    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    im = axes[1].imshow(attr_np, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    axes[2].imshow(image)
    axes[2].imshow(attr_np, cmap='jet', alpha=0.5)
    axes[2].set_title(f"Overlay\nPrediction: {CLASS_NAMES[pred_class]}", 
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    result_image = Image.open(buf)
    plt.close(fig)
    
    # Prediction text
    prediction_text = f"**Prediction:** {CLASS_NAMES[pred_class]}\n\n"
    prediction_text += f"**Confidence:** {confidence*100:.2f}%\n\n"
    prediction_text += "**Top 3 Predictions:**\n"
    
    top3_probs, top3_indices = torch.topk(probabilities[0], 3)
    for prob, idx in zip(top3_probs, top3_indices):
        prediction_text += f"- {CLASS_NAMES[idx]}: {prob.item()*100:.2f}%\n"
    
    return prediction_text, result_image

# Gradio Interface
with gr.Blocks(title="🔍 Explainable Image Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 Explainable Image Classifier with Grad-CAM
    
    Upload an image and see:
    - **What** the AI predicts (classification)
    - **Why** it made that decision (Grad-CAM visualization)
    
    **Supported categories:** airplane, car, bird, cat, deer, dog, frog, horse, ship, truck
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            predict_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
        
        with gr.Column():
            output_text = gr.Markdown(label="Prediction Results")
            output_image = gr.Image(label="Grad-CAM Visualization", type="pil")
    
    predict_btn.click(
        fn=predict_and_explain,
        inputs=input_image,
        outputs=[output_text, output_image]
    )
    
    gr.Markdown("""
    ---
    ### 🧠 About This Model
    - **Architecture:** ResNet18 (transfer learning)
    - **Training Data:** CIFAR-10 (60,000 images)
    - **Explainability:** Grad-CAM visualization
    """)

if __name__ == "__main__":
    demo.launch()