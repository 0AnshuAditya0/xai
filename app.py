

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr
from captum.attr import LayerGradCam
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import json, urllib.request
from torch.nn.functional import interpolate
import warnings
warnings.filterwarnings('ignore')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {DEVICE}")


@torch.no_grad()
def load_model_and_labels():
    print("Loading ResNet50...")
    model = models.resnet50(weights='IMAGENET1K_V2')
    model.eval()
    model = model.to(DEVICE)

    print("Loading ImageNet labels...")
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = json.loads(urllib.request.urlopen(url).read())
    print(f"Model ready – {len(labels)} classes")
    return model, labels

model, IMAGENET_LABELS = load_model_and_labels()


target_layer = model.layer4[-1]
gradcam = LayerGradCam(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])


def predict_and_explain(image):
    if image is None:
        return "Please upload an image", None

    try:
        
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            top5_prob, top5_idx = torch.topk(probabilities, 5)
        
        pred_class = top5_idx[0][0].item()
        confidence = top5_prob[0][0].item()

        
        attributions = gradcam.attribute(img_tensor, target=pred_class)
        attr_resized = interpolate(attributions, size=(224, 224), mode='bilinear', align_corners=False)
        attr_np = attr_resized.squeeze().cpu().detach().numpy()
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)

       
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.patch.set_facecolor('#0a0a0a')
        plt.subplots_adjust(wspace=0.12, left=0.02, right=0.98, top=0.92, bottom=0.08)

        
        axes[0].imshow(image)
        axes[0].set_title("Original", fontsize=13, fontweight='600', color='#e0e0e0', pad=12)
        axes[0].axis('off')

        
        im = axes[1].imshow(attr_np, cmap='viridis')
        axes[1].set_title("Grad-CAM", fontsize=13, fontweight='600', color='#e0e0e0', pad=12)
        axes[1].axis('off')
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9, colors='#a0a0a0')
        cbar.set_label('Focus', rotation=270, labelpad=15, color='#a0a0a0', fontsize=10)

       
        axes[2].imshow(image)
        axes[2].imshow(attr_np, cmap='viridis', alpha=0.5)
        axes[2].set_title("AI Focus", fontsize=13, fontweight='600', color='#e0e0e0', pad=12)
        axes[2].axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0a0a0a')
        buf.seek(0)
        result_image = Image.open(buf)
        plt.close(fig)

       
        if confidence > 0.8:
            badge = "high"
            badge_text = "High Confidence"
            badge_icon = "🎯"
        elif confidence > 0.5:
            badge = "medium"
            badge_text = "Medium Confidence"
            badge_icon = "⚡"
        else:
            badge = "low"
            badge_text = "Low Confidence"
            badge_icon = "⚠️"

        
        top5_html = "<div class='top5-grid'>"
        icons = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        for i, (prob, idx) in enumerate(zip(top5_prob[0], top5_idx[0])):
            pct = prob.item() * 100
            top5_html += f"""
            <div class='top5-row'>
                <span class='rank'>{icons[i]}</span>
                <span class='label'>{IMAGENET_LABELS[idx.item()].title()}</span>
                <div class='bar-wrap'><div class='bar' style='width:{pct}%'></div></div>
                <span class='pct'>{pct:.1f}%</span>
            </div>
            """
        top5_html += "</div>"

        
        prediction_text = f"""
<div class="result-card">
    <div class="pred-header">
        <h2 class="pred-label">{IMAGENET_LABELS[pred_class].title()}</h2>
        <div class="badge badge-{badge}">{badge_icon} {badge_text}</div>
    </div>
    <div class="conf-score">{confidence*100:.1f}%</div>
    <div class="divider"></div>
    {top5_html}
</div>
"""
        return prediction_text, result_image

    except Exception as e:
        return f"<div class='error-msg'>⚠️ Error: {str(e)}</div>", None


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { 
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body, .gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    width: 100vw !important;
    min-height: 100vh !important;
    max-width: 100vw !important;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e0e0e0 !important;
    overflow-x: hidden !important;
}

.gradio-container { padding: 0 !important; }

body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0.03;
    pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' /%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' /%3E%3C/svg%3E");
    z-index: 1;
}

.main-wrapper {
    padding: 1.5rem;
    max-width: 1920px;
    margin: 0 auto;
    position: relative;
    z-index: 2;
}

.hero-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
}

.hero-header h1 {
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #3b82f6 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem;
    letter-spacing: -1px;
}

.hero-header .subtitle {
    font-size: clamp(0.95rem, 2vw, 1.2rem);
    color: #808080;
    font-weight: 400;
    margin: 0;
}

.tab-nav {
    display: flex;
    gap: 2rem;
    justify-content: center;
    margin-bottom: 2rem;
    padding: 0.5rem;
    background: rgba(20, 20, 20, 0.6);
    border-radius: 16px;
    width: fit-content;
    margin-left: auto;
    margin-right: auto;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.gr-tab {
    background: transparent !important;
    border: none !important;
    border-radius: 12px !important;
    color: #a0a0a0 !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    transition: all 0.3s ease !important;
    font-size: 1rem !important;
    cursor: pointer !important;
}

.gr-tab:hover {
    background: rgba(59, 130, 246, 0.1) !important;
    color: #e0e0e0 !important;
}

.gr-tab.selected {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
}

.top-section {
    display: grid;
    grid-template-columns: 400px 1fr;
    gap: 1.25rem;
    margin-bottom: 1.25rem;
}

.glass-card {
    background: rgba(20, 20, 20, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 1.5rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
}

.glass-card:hover {
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
    transform: translateY(-2px);
}

.upload-panel {
    background: rgba(20, 20, 20, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 1.5rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    height: fit-content;
}

.section-label {
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 1rem;
    text-align: center;
    letter-spacing: 0.5px;
}

#input-image {
    border: 2px dashed rgba(59, 130, 246, 0.4) !important;
    border-radius: 20px !important;
    background: rgba(10, 10, 10, 0.6) !important;
    height: 320px !important;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

#input-image::before {
    display: none;
}

#input-image:hover {
    border-color: #3b82f6 !important;
    background: rgba(20, 20, 30, 0.8) !important;
    transform: scale(1.02);
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.2);
}

.btn-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 1rem;
}

.gr-button {
    border-radius: 14px !important;
    font-weight: 700 !important;
    height: 50px !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    border: none !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.gr-button-primary {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4) !important;
}

.gr-button-primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(59, 130, 246, 0.6) !important;
}

.gr-button-secondary {
    background: rgba(40, 40, 40, 0.8) !important;
    color: #a0a0a0 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.gr-button-secondary:hover {
    background: rgba(50, 50, 50, 0.9) !important;
    color: #e0e0e0 !important;
}

.results-panel {
    background: rgba(20, 20, 20, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 1.75rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.result-card {
    padding: 0;
}

.pred-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
    margin-bottom: 0.75rem;
}

.pred-label {
    font-size: clamp(1.5rem, 3vw, 2rem);
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}

.badge {
    padding: 0.5rem 1.25rem;
    border-radius: 50px;
    font-size: 0.875rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.badge-high { 
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
}
.badge-medium { 
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
}
.badge-low { 
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
}

.conf-score {
    font-size: clamp(2rem, 5vw, 3rem);
    font-weight: 900;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1.25rem;
    letter-spacing: -1px;
}

.divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
    margin: 1.5rem 0;
}

.top5-grid {
    display: flex;
    flex-direction: column;
    gap: 0.875rem;
}

.top5-row {
    display: grid;
    grid-template-columns: 40px 1fr auto 70px;
    align-items: center;
    gap: 0.875rem;
    font-size: 0.95rem;
    padding: 0.5rem;
    border-radius: 12px;
    background: rgba(30, 30, 30, 0.5);
    transition: all 0.3s ease;
}

.top5-row:hover {
    background: rgba(40, 40, 40, 0.7);
    transform: translateX(5px);
}

.rank {
    font-size: 1.5rem;
    text-align: center;
}

.label {
    color: #e0e0e0;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.bar-wrap {
    background: rgba(40, 40, 40, 0.8);
    height: 10px;
    border-radius: 5px;
    overflow: hidden;
    min-width: 100px;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3);
}

.bar {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    height: 100%;
    transition: width 1s ease;
    border-radius: 5px;
    box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
}

.pct {
    color: #3b82f6;
    font-weight: 700;
    font-size: 0.9rem;
    text-align: right;
}

.viz-section {
    background: rgba(20, 20, 20, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 1.75rem;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

#result-image {
    border-radius: 16px !important;
    overflow: hidden;
    width: 100%;
    height: auto;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
}

.about-content {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
    background: rgba(20, 20, 20, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    line-height: 1.8;
}

.placeholder {
    text-align: center;
    padding: 4rem 1.5rem;
    color: #606060;
    font-size: 1.1rem;
    line-height: 1.6;
}

.placeholder strong {
    color: #3b82f6;
}

.error-msg {
    color: #ef4444;
    background: rgba(239, 68, 68, 0.1);
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.gr-accordion {
    background: rgba(20, 20, 20, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(20px);
    margin-top: 1.5rem;
    overflow: hidden;
}

.gr-accordion summary {
    color: #e0e0e0 !important;
    font-weight: 700 !important;
    padding: 1.25rem 1.5rem !important;
    font-size: 1.1rem !important;
    cursor: pointer;
    transition: all 0.3s ease;
}

.gr-accordion summary:hover {
    background: rgba(59, 130, 246, 0.1);
}

.gr-accordion[open] summary {
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.gr-tab {
    background: rgba(20, 20, 20, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 16px 16px 0 0 !important;
    color: #a0a0a0 !important;
    font-weight: 600 !important;
    transition: all 0.3s ease;
}

.gr-tab:hover {
    background: rgba(30, 30, 30, 0.8) !important;
    color: #e0e0e0 !important;
}

.gr-tab.selected {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border-bottom: none !important;
}

.gr-tabitem {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

footer, .footer { display: none !important; }

::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: rgba(20, 20, 20, 0.5);
}

::-webkit-scrollbar-thumb {
    background: rgba(59, 130, 246, 0.5);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(59, 130, 246, 0.7);
}

@media (max-width: 1024px) {
    .top-section {
        grid-template-columns: 360px 1fr;
        gap: 1rem;
    }
    
    #input-image {
        height: 280px !important;
    }
    
    .features-grid {
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    }
}

@media (max-width: 768px) {
    .main-wrapper {
        padding: 1rem;
    }
    
    .hero-header {
        padding: 1.5rem 0.75rem;
    }
    
    .top-section {
        grid-template-columns: 1fr;
        gap: 1rem;
    }
    
    .upload-panel, .results-panel, .viz-section {
        width: 100%;
    }
    
    #input-image {
        height: 240px !important;
    }
    
    .top5-row {
        grid-template-columns: 35px 1fr 60px;
        gap: 0.625rem;
    }
    
    .bar-wrap {
        grid-column: 1 / -1;
        margin-top: 0.375rem;
    }
    
    .btn-row {
        flex-direction: column;
    }
    
    .gr-button {
        width: 100% !important;
    }
    
    .about-hero {
        padding: 2rem 1rem;
    }
    
    .use-case-item {
        flex-direction: column;
        gap: 1rem;
    }
}

@media (max-width: 480px) {
    .hero-header h1 {
        font-size: 1.75rem;
    }
    
    #input-image {
        height: 220px !important;
    }
    
    .pred-header {
        flex-direction: column;
        align-items: flex-start;
    }
    
    .gr-button {
        height: 46px !important;
        font-size: 0.875rem !important;
    }
    
    .feature-card {
        padding: 1.5rem;
    }
    
    .use-cases {
        padding: 1.5rem;
    }
}
"""


with gr.Blocks(css=custom_css, theme=gr.themes.Base(), title="Explainable AI Classifier") as demo:

    
    gr.HTML("""
    <link rel="icon" href="https://res.cloudinary.com/ddn0xuwut/image/upload/v1761284764/encryption_hc0fxo.png" type="image/png">
    """)

    with gr.Tabs() as tabs:
        
        with gr.TabItem("🔮 Classifier", id=0):
            with gr.Column(elem_classes="main-wrapper"):
                gr.HTML("""
                <div class="hero-header">
                    <h1>Explainable AI Classifier</h1>
                    <p class="subtitle">See exactly what the AI sees – powered by ResNet50 + Grad-CAM</p>
                </div>
                """)

            
            with gr.Row(elem_classes="top-section"):
                
                with gr.Column(scale=0, min_width=400, elem_classes="upload-panel"):
                    gr.HTML("<div class='section-label'>📤 Upload Image</div>")
                    
                    input_image = gr.Image(
                        type="pil",
                        label=None,
                        elem_id="input-image",
                        show_label=False,
                        container=False
                    )
                    
                    with gr.Row(elem_classes="btn-row"):
                        predict_btn = gr.Button("🚀 Analyze", variant="primary", size="lg", scale=2)
                        clear_btn = gr.ClearButton([input_image], value="🗑️ Clear", size="lg", scale=1)
                    
                    with gr.Accordion("💡 Tips for Best Results", open=False):
                        gr.Markdown("""
                        - Use **clear, well-lit photos**  
                        - **Center the main subject**  
                        - Avoid heavy filters or edits  
                        - Works best with **single objects**
                        - Supports JPG, PNG, WebP formats
                        """)

                
                with gr.Column(scale=1, elem_classes="results-panel"):
                    output_text = gr.HTML(
                        """<div class='placeholder'>
                        <strong>👋 Welcome!</strong><br><br>
                        Upload an image and click <strong>Analyze</strong> to see:<br>
                        ✓ AI predictions with confidence scores<br>
                        ✓ Visual explanations (what the AI focuses on)<br>
                        ✓ Top 5 possible classifications
                        </div>"""
                    )

            
            with gr.Column(elem_classes="viz-section"):
                gr.HTML("<div class='section-label'>🎯 Visual Explainability</div>")
                output_image = gr.Image(
                    label=None,
                    type="pil",
                    show_label=False,
                    elem_id="result-image",
                    height=450
                )

            with gr.Accordion("📚 How It Works", open=False):
                gr.Markdown("""
                ### The Technology Behind This Tool
                
                **1. ResNet50 Neural Network**  
                A deep learning model trained on 1,000+ categories from ImageNet dataset.  
                Achieves 76%+ accuracy on real-world images.
                
                **2. Grad-CAM (Gradient-weighted Class Activation Mapping)**  
                Developed by researchers at Georgia Tech, Grad-CAM highlights which regions  
                of the image were most important for the AI's decision.
                
                **3. Real-Time Inference**  
                Your image is processed locally in this interface – we never store your uploads.
                
                ### Why This Matters
                - **Transparency**: See *why* the AI made its decision
                - **Trust**: Verify the AI is focusing on relevant features
                - **Debugging**: Understand when and why predictions fail
                - **Compliance**: Meet explainability requirements (EU AI Act, GDPR)
                """)

    
        with gr.TabItem("ℹ️ About", id=1):
            with gr.Column(elem_classes="main-wrapper about-page"):
                gr.HTML("""
                <div class="about-hero">
                    <h1>About Explainable AI</h1>
                    <p>Making artificial intelligence transparent, interpretable, and trustworthy for everyone</p>
                </div>
                """)
                
                gr.HTML("""
                <div class="features-grid">
                    <div class="feature-card">
                        <span class="feature-icon">🔍</span>
                        <h3>Transparency</h3>
                        <p>No more "black box" decisions. See exactly which parts of your image influenced the AI's prediction with visual heatmaps.</p>
                    </div>
                    
                    <div class="feature-card">
                        <span class="feature-icon">🛡️</span>
                        <h3>Trust & Safety</h3>
                        <p>Verify that AI models are making decisions based on relevant features, not spurious correlations or biases.</p>
                    </div>
                    
                    <div class="feature-card">
                        <span class="feature-icon">⚖️</span>
                        <h3>Regulatory Compliance</h3>
                        <p>Meet requirements from EU AI Act, GDPR, FDA guidelines for medical AI, and other global regulations.</p>
                    </div>
                    
                    <div class="feature-card">
                        <span class="feature-icon">🎓</span>
                        <h3>Educational</h3>
                        <p>Learn how deep learning works by visualizing what neural networks "see" when processing images.</p>
                    </div>
                    
                    <div class="feature-card">
                        <span class="feature-icon">🚀</span>
                        <h3>Production-Ready</h3>
                        <p>Built with PyTorch and Captum (Meta's interpretability library), this tool uses industry-standard techniques.</p>
                    </div>
                    
                    <div class="feature-card">
                        <span class="feature-icon">🌐</span>
                        <h3>Open Source Spirit</h3>
                        <p>Based on open research and publicly available models. Knowledge should be accessible to all.</p>
                    </div>
                </div>
                
                <div class="use-cases">
                    <h2>Real-World Applications</h2>
                    
                    <div class="use-case-item">
                        <span class="use-case-icon">🏥</span>
                        <div class="use-case-content">
                            <h3>Healthcare & Medical Imaging</h3>
                            <p><strong>Problem:</strong> Radiologists need to trust AI diagnostic tools.<br>
                            <strong>Solution:</strong> Grad-CAM shows if the AI is focusing on tumors, lesions, or irrelevant artifacts. This builds trust and helps doctors make informed decisions.</p>
                        </div>
                    </div>
                    
                    <div class="use-case-item">
                        <span class="use-case-icon">🚗</span>
                        <div class="use-case-content">
                            <h3>Autonomous Vehicles</h3>
                            <p><strong>Problem:</strong> Self-driving cars must explain their perception to passengers and regulators.<br>
                            <strong>Solution:</strong> Visual explanations confirm the AI detected pedestrians, traffic signs, and road boundaries – not random patterns.</p>
                        </div>
                    </div>
                    
                    <div class="use-case-item">
                        <span class="use-case-icon">💰</span>
                        <div class="use-case-content">
                            <h3>Financial Services</h3>
                            <p><strong>Problem:</strong> Banks must explain loan approval/denial decisions (Fair Lending Act).<br>
                            <strong>Solution:</strong> Explainable AI shows which factors (income, credit score, employment) influenced the decision, ensuring fairness.</p>
                        </div>
                    </div>
                    
                    <div class="use-case-item">
                        <span class="use-case-icon">🔒</span>
                        <div class="use-case-content">
                            <h3>Security & Surveillance</h3>
                            <p><strong>Problem:</strong> Facial recognition and threat detection systems face scrutiny over bias.<br>
                            <strong>Solution:</strong> Grad-CAM reveals if the system is using facial features appropriately or making decisions based on background elements.</p>
                        </div>
                    </div>
                    
                    <div class="use-case-item">
                        <span class="use-case-icon">🌾</span>
                        <div class="use-case-content">
                            <h3>Agriculture & Environment</h3>
                            <p><strong>Problem:</strong> Farmers need to verify AI crop disease detection accuracy.<br>
                            <strong>Solution:</strong> Visual explanations show if the AI identified actual disease symptoms (leaf discoloration, spots) or environmental factors.</p>
                        </div>
                    </div>
                    
                    <div class="use-case-item">
                        <span class="use-case-icon">🛒</span>
                        <div class="use-case-content">
                            <h3>E-Commerce & Retail</h3>
                            <p><strong>Problem:</strong> Product recognition systems must be reliable for inventory and checkout.<br>
                            <strong>Solution:</strong> Explainability helps debug failures – did the AI misidentify a product because of lighting, angle, or packaging?</p>
                        </div>
                    </div>
                </div>
                """)
                
                gr.Markdown("""
                ## Why Explainability is the Future
                
                As AI becomes more integrated into critical decision-making processes, **explainability is no longer optional**:
                
                - **EU AI Act (2024)**: High-risk AI systems must provide explanations
                - **GDPR Right to Explanation**: Users can demand explanations for automated decisions
                - **FDA Guidelines**: Medical AI must demonstrate interpretability
                - **Ethical AI Standards**: Leading tech companies committed to transparency
                
                ---
                
                ## Technical Stack
                
                This tool is built with:
                - **PyTorch**: Deep learning framework
                - **torchvision**: Pre-trained ResNet50 model
                - **Captum**: Meta's interpretability library
                - **Gradio**: Interactive web interface
                - **Grad-CAM**: Visual explanation technique
                
                ---
                
                
                """)


    predict_btn.click(
        fn=predict_and_explain,
        inputs=[input_image],
        outputs=[output_text, output_image]
    )


if __name__ == "__main__":
    demo.launch(share=False, show_error=True)