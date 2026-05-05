#  Explainable AI: Bone Fracture Triage

This repository contains the source code and deployment dashboard for my Master's research project at Ulster University. The project utilizes a custom-trained **EfficientNet-B0** convolutional neural network to detect musculoskeletal abnormalities in X-ray images.

To bridge the "Black Box" trust gap in medical AI, this pipeline integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide real-time visual interpretability for clinicians.

### Repository Contents
* `Google_Colab_Notebook.ipynb` - The complete PyTorch training pipeline, data augmentation, and evaluation metrics.
* `app.py` - The Streamlit web dashboard featuring real-time inference and Grad-CAM generation.
* `mura_efficientnet_10epochs.pth` - The trained model weights.

### How to Run the Dashboard Locally
1. Clone this repository.
2. Install the required dependencies: `pip install streamlit torch torchvision pillow numpy opencv-python-headless`
3. Run the application: `streamlit run app.py`
