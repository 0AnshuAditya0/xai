🔍 Explainable AI Image Classifier (XAI)

An Explainable AI Image Classification Web App powered by ResNet50 + Grad-CAM.
See exactly what the AI focuses on when making predictions — transparent & trustworthy AI.

✨ Features

✅ Image Classification on 1000 ImageNet categories
✅ Explainability using Gradient-weighted Class Activation Mapping (Grad-CAM)
✅ Interactive & Fast UI built with Gradio
✅ GPU Support for real-time inference
✅ Production-ready deployment design

🧠 How It Works

Input image → Preprocessing (ImageNet transforms)

Model: ResNet50 (pretrained on 1.2M ImageNet images)

AI prediction → Top-5 result visualization

Grad-CAM highlights important pixel regions the model uses to decide ✅
Helps detect:

Model biases

Wrong object focus

Reliability of prediction

🏗 Tech Stack
Component	Technology
Model	ResNet50 (TorchVision)
Explainability	Grad-CAM via Captum
Interface	Gradio
Backend	PyTorch
Deployment	Local / Hugging Face / Cloud
📦 Installation
git clone https://github.com/your-username/xai.git
cd xai
pip install -r requirements.txt
python app.py


✅ Then open the local URL and upload any image!

📌 Requirements

Python 3.9+

PyTorch 2.0+

Gradio 3.0+

CUDA GPU (optional but faster)

🚀 Deployment Ready

You can deploy on:

Hugging Face Spaces

AWS / GCP / Azure

Local / On-prem

🔗 Add your deployment link here:

➡️ Live Demo: Coming Soon
➡️ GitHub Repo: https://github.com/0AnshuAditya0/xai

🎯 Real-World Impact

Perfect for industries where AI must be explainable, such as:

🏥 Healthcare (X-ray / MRI model audits)

🚗 Autonomous vehicle perception validation

📚 AI education and learning

🔒 Identity & security checks

⚖️ Fairness auditing in AI systems

🙌 Author

Anshu Aditya
AI Engineer — Building transparent & responsible AI 🚀
Let’s connect: Add your LinkedIn/GitHub/Twitter here

📝 License

MIT License
Free for research, learning & production use ✅