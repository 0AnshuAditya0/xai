"""
Explainable AI Image Classifier - Optimized Production Version
===============================================================
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from captum.attr import LayerGradCam
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import urllib.request
from torch.nn.functional import interpolate
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & MODEL LOADING
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def load_model_and_labels():
    """Load ResNet50 with optimized settings"""
    model = models.resnet50(weights='IMAGENET1K_V2')
    model.eval()
    model = model.to(DEVICE)
    
    # Load cleaner ImageNet labels
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = urllib.request.urlopen(url)
    labels = [line.decode('utf-8').strip() for line in response.readlines()]
    
    return model, labels

model, IMAGENET_LABELS = load_model_and_labels()

# Setup Grad-CAM
target_layer = model.layer4[-1]
gradcam = LayerGradCam(model, target_layer)

# Optimized preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# OPTIMIZED PREDICTION FUNCTION
# ============================================================================

def predict_and_explain(image):
    if image is None:
        return "Please upload an image", None, None

    try:
        # Preprocess
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Predict with temperature scaling for better confidence
        with torch.no_grad():
            output = model(img_tensor)
            # Apply temperature scaling (makes confidence more realistic)
            temperature = 1.0  # Adjust between 0.8-1.5 if needed
            scaled_output = output / temperature
            probabilities = torch.softmax(scaled_output, dim=1)
            top10_prob, top10_idx = torch.topk(probabilities, 10)
        
        pred_class = top10_idx[0][0].item()
        confidence = top10_prob[0][0].item()

        # Generate Grad-CAM
        attributions = gradcam.attribute(img_tensor, target=pred_class)
        attr_resized = interpolate(attributions, size=(224, 224), mode='bilinear', align_corners=False)
        attr_np = attr_resized.squeeze().cpu().detach().numpy()
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)

        # Create enhanced visualization (LARGER SIZE)
        fig = plt.figure(figsize=(20, 12))  # Increased from 18x10 to 20x12
        fig.patch.set_facecolor('#0a0a0a')
        
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.25, wspace=0.12)  # More space for images
        
        # Top row: Larger image panels
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, :])
        
        # Panel 1: Original
        ax1.imshow(image)
        ax1.set_title("Original Image", fontsize=15, fontweight='600', color='#e0e0e0', pad=15)
        ax1.axis('off')
        
        # Panel 2: Detailed Heatmap
        im = ax2.imshow(attr_np, cmap='jet', interpolation='bilinear')
        ax2.set_title("Grad-CAM Heatmap", fontsize=15, fontweight='600', color='#e0e0e0', pad=15)
        ax2.axis('off')
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10, colors='#a0a0a0')
        cbar.set_label('Importance', rotation=270, labelpad=20, color='#e0e0e0', fontsize=11, fontweight='600')
        
        # Panel 3: Overlay
        ax3.imshow(image)
        ax3.imshow(attr_np, cmap='jet', alpha=0.5, interpolation='bilinear')
        ax3.set_title(f"AI Focus: {IMAGENET_LABELS[pred_class]}", fontsize=15, fontweight='600', color='#e0e0e0', pad=15)
        ax3.axis('off')
        
        # Panel 4: Bar Graph
        top10_labels = [IMAGENET_LABELS[idx.item()] for idx in top10_idx[0]]
        top10_probs = [prob.item() * 100 for prob in top10_prob[0]]
        
        colors = ['#10b981' if i == 9 else '#3b82f6' if i >= 7 else '#8b5cf6' for i in range(10)]
        bars = ax4.barh(range(10), top10_probs[::-1], color=colors[::-1], edgecolor='#1a1a1a', linewidth=1.5)
        
        ax4.set_yticks(range(10))
        ax4.set_yticklabels(top10_labels[::-1], fontsize=12, color='#e0e0e0')
        ax4.set_xlabel('Confidence (%)', fontsize=13, color='#e0e0e0', fontweight='600')
        ax4.set_title('Top 10 Predictions', fontsize=16, fontweight='700', color='#e0e0e0', pad=15)
        ax4.set_xlim([0, 100])
        ax4.grid(axis='x', alpha=0.2, color='#404040', linestyle='--')
        ax4.set_facecolor('#0a0a0a')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.spines['left'].set_color('#404040')
        ax4.spines['bottom'].set_color('#404040')
        ax4.tick_params(colors='#a0a0a0', labelsize=11)
        
        # Add percentage labels
        for bar, prob in zip(bars, top10_probs[::-1]):
            ax4.text(prob + 1.5, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.1f}%', va='center', fontsize=11, color='#e0e0e0', fontweight='600')
        
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='#0a0a0a')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close(fig)

        # Create detailed heatmap panel
        fig2, axes = plt.subplots(1, 3, figsize=(18, 6))  # Increased from 15x5 to 18x6
        fig2.patch.set_facecolor('#0a0a0a')
        
        axes[0].imshow(image)
        axes[0].set_title("Original", fontsize=14, fontweight='600', color='#e0e0e0', pad=12)
        axes[0].axis('off')
        
        im2 = axes[1].imshow(attr_np, cmap='viridis', interpolation='gaussian')
        axes[1].set_title("High-Res Heatmap", fontsize=14, fontweight='600', color='#e0e0e0', pad=12)
        axes[1].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=10, colors='#a0a0a0')
        
        axes[2].imshow(image)
        axes[2].imshow(attr_np, cmap='hot', alpha=0.6, interpolation='bilinear')
        axes[2].contour(attr_np, levels=5, colors='white', linewidths=1.5, alpha=0.8)
        axes[2].set_title("Contour Analysis", fontsize=14, fontweight='600', color='#e0e0e0', pad=12)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        buf2 = BytesIO()
        plt.savefig(buf2, format='png', dpi=110, bbox_inches='tight', facecolor='#0a0a0a')
        buf2.seek(0)
        detailed_heatmap = Image.open(buf2)
        plt.close(fig2)

        # Generate HTML output
        badge = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        badge_text = "High Confidence" if confidence > 0.8 else "Medium Confidence" if confidence > 0.5 else "Low Confidence"
        badge_icon = "🎯" if confidence > 0.8 else "⚡" if confidence > 0.5 else "⚠️"

        top5_html = "<div class='top5-grid'>"
        icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        for i, (prob, idx) in enumerate(zip(top10_prob[0][:5], top10_idx[0][:5])):
            pct = prob.item() * 100
            top5_html += f"""
            <div class='top5-row'>
                <span class='rank'>{icons[i]}</span>
                <span class='label'>{IMAGENET_LABELS[idx.item()]}</span>
                <div class='bar-wrap'><div class='bar' style='width:{pct}%'></div></div>
                <span class='pct'>{pct:.2f}%</span>
            </div>"""
        top5_html += "</div>"

        prediction_text = f"""
<div class="result-card">
    <div class="pred-header">
        <h2 class="pred-label">{IMAGENET_LABELS[pred_class]}</h2>
        <div class="badge badge-{badge}">{badge_icon} {badge_text}</div>
    </div>
    <div class="conf-score">{confidence*100:.2f}%</div>
    <div class="divider"></div>
    {top5_html}
</div>"""
        
        return prediction_text, result_image, detailed_heatmap

    except Exception as e:
        return f"<div class='error-msg'>⚠️ Error: {str(e)}</div>", None, None

# ============================================================================
# STREAMLINED CSS
# ============================================================================

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
body, .gradio-container { margin: 0 !important; padding: 0 !important; width: 100vw !important; min-height: 100vh !important; max-width: 100vw !important; background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%) !important; font-family: 'Inter', sans-serif !important; color: #e0e0e0 !important; overflow-x: hidden !important; }
.gradio-container { padding: 0 !important; }
.main-wrapper { padding: 1.5rem; max-width: 1920px; margin: 0 auto; position: relative; z-index: 2; }
.hero-header { text-align: center; padding: 2rem 1rem 1.5rem; margin-bottom: 1.5rem; }
.hero-header h1 { font-size: clamp(2rem, 5vw, 3.5rem); font-weight: 800; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 0.5rem; letter-spacing: -1px; }
.hero-header .subtitle { font-size: clamp(0.95rem, 2vw, 1.2rem); color: #808080; font-weight: 400; margin: 0; }
.top-section { display: grid; grid-template-columns: 400px 1fr; gap: 1.25rem; margin-bottom: 1.25rem; }
.upload-panel, .results-panel, .viz-section { background: rgba(20, 20, 20, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 24px; padding: 1.5rem; backdrop-filter: blur(20px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4); }
.section-label { font-size: 1.1rem; font-weight: 700; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 1rem; text-align: center; letter-spacing: 0.5px; }
#input-image { border: 2px dashed rgba(59, 130, 246, 0.4) !important; border-radius: 20px !important; background: rgba(10, 10, 10, 0.6) !important; height: 320px !important; transition: all 0.3s ease; }
#input-image:hover { border-color: #3b82f6 !important; background: rgba(20, 20, 30, 0.8) !important; transform: scale(1.02); box-shadow: 0 0 30px rgba(59, 130, 246, 0.2); }
.btn-row { display: flex; gap: 0.75rem; margin-top: 1rem; }
.gr-button { border-radius: 14px !important; font-weight: 700 !important; height: 50px !important; font-size: 0.95rem !important; transition: all 0.3s ease !important; border: none !important; letter-spacing: 0.5px; text-transform: uppercase; }
.gr-button-primary { background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important; color: white !important; box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4) !important; }
.gr-button-primary:hover { transform: translateY(-3px) !important; box-shadow: 0 8px 30px rgba(59, 130, 246, 0.6) !important; }
.gr-button-secondary { background: rgba(40, 40, 40, 0.8) !important; color: #a0a0a0 !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; }
.pred-header { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem; margin-bottom: 0.75rem; }
.pred-label { font-size: clamp(1.5rem, 3vw, 2rem); font-weight: 800; color: #ffffff; margin: 0; letter-spacing: -0.5px; }
.badge { padding: 0.5rem 1.25rem; border-radius: 50px; font-size: 0.875rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); }
.badge-high { background: linear-gradient(135deg, #10b981, #059669); color: white; }
.badge-medium { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
.badge-low { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
.conf-score { font-size: clamp(2rem, 5vw, 3rem); font-weight: 900; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1.25rem; letter-spacing: -1px; }
.divider { height: 2px; background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent); margin: 1.5rem 0; }
.top5-grid { display: flex; flex-direction: column; gap: 0.875rem; }
.top5-row { display: grid; grid-template-columns: 40px 1fr auto 80px; align-items: center; gap: 0.875rem; font-size: 0.95rem; padding: 0.5rem; border-radius: 12px; background: rgba(30, 30, 30, 0.5); transition: all 0.3s ease; }
.top5-row:hover { background: rgba(40, 40, 40, 0.7); transform: translateX(5px); }
.rank { font-size: 1.5rem; text-align: center; }
.label { color: #e0e0e0; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-wrap { background: rgba(40, 40, 40, 0.8); height: 10px; border-radius: 5px; overflow: hidden; min-width: 100px; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3); }
.bar { background: linear-gradient(90deg, #3b82f6, #8b5cf6); height: 100%; transition: width 1s ease; border-radius: 5px; box-shadow: 0 0 10px rgba(59, 130, 246, 0.5); }
.pct { color: #3b82f6; font-weight: 700; font-size: 0.9rem; text-align: right; }
#result-image, #detailed-heatmap { border-radius: 16px !important; overflow: hidden; width: 100%; height: auto; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5); }
.placeholder { text-align: center; padding: 4rem 1.5rem; color: #606060; font-size: 1.1rem; line-height: 1.6; }
.placeholder strong { color: #3b82f6; }
.error-msg { color: #ef4444; background: rgba(239, 68, 68, 0.1); padding: 1.5rem; border-radius: 16px; text-align: center; border: 1px solid rgba(239, 68, 68, 0.3); }
.gr-accordion { background: rgba(20, 20, 20, 0.8) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 20px !important; margin-top: 1.5rem; }
.gr-accordion summary { color: #e0e0e0 !important; font-weight: 700 !important; padding: 1.25rem 1.5rem !important; font-size: 1.1rem !important; }
footer, .footer { display: none !important; }
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: rgba(20, 20, 20, 0.5); }
::-webkit-scrollbar-thumb { background: rgba(59, 130, 246, 0.5); border-radius: 5px; }
@media (max-width: 768px) { .top-section { grid-template-columns: 1fr; } #input-image { height: 240px !important; } .top5-row { grid-template-columns: 35px 1fr 70px; } .bar-wrap { grid-column: 1 / -1; margin-top: 0.375rem; } }
"""

# ============================================================================
# STREAMLINED INTERFACE
# ============================================================================

with gr.Blocks(css=custom_css, theme=gr.themes.Base(), title="Explainable AI") as demo:
    gr.HTML('<link rel="icon" href="https://res.cloudinary.com/ddn0xuwut/image/upload/v1761284764/encryption_hc0fxo.png" type="image/png">')

    with gr.Column(elem_classes="main-wrapper"):
        gr.HTML('<div class="hero-header"><h1>Explainable AI Classifier</h1><p class="subtitle">See exactly what the AI sees – powered by ResNet50 + Grad-CAM</p></div>')

        with gr.Row(elem_classes="top-section"):
            with gr.Column(scale=0, min_width=400, elem_classes="upload-panel"):
                gr.HTML("<div class='section-label'>📤 Upload Image</div>")
                input_image = gr.Image(type="pil", label=None, elem_id="input-image", show_label=False, container=False)
                with gr.Row(elem_classes="btn-row"):
                    predict_btn = gr.Button("🚀 Analyze", variant="primary", size="lg", scale=2)
                    clear_btn = gr.ClearButton([input_image], value="🗑️ Clear", size="lg", scale=1)

            with gr.Column(scale=1, elem_classes="results-panel"):
                output_text = gr.HTML('<div class="placeholder"><strong>👋 Welcome!</strong><br><br>Upload an image and click <strong>Analyze</strong></div>')

        with gr.Column(elem_classes="viz-section"):
            gr.HTML("<div class='section-label'>🎯 Visual Explainability (Includes Bar Graph)</div>")
            output_image = gr.Image(label=None, type="pil", show_label=False, elem_id="result-image", height=700)

        with gr.Column(elem_classes="viz-section"):
            gr.HTML("<div class='section-label'>🔬 Advanced Heatmap Analysis</div>")
            detailed_heatmap = gr.Image(label=None, type="pil", show_label=False, elem_id="detailed-heatmap", height=500)

    predict_btn.click(fn=predict_and_explain, inputs=[input_image], outputs=[output_text, output_image, detailed_heatmap])

if __name__ == "__main__":
    demo.launch(share=False, show_error=True)