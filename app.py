import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# ==========================================
# 1. PAGE SETUP & UI STYLING
# ==========================================
st.set_page_config(page_title="Bone Fracture AI Triage", page_icon="🦴", layout="wide")

st.title("🦴 Explainable AI: Bone Fracture Triage")
st.markdown("""
Welcome to the AI Radiography Dashboard. Upload a musculoskeletal X-ray to receive an instant 
prediction powered by EfficientNet-B0, complete with real-time Grad-CAM spatial interpretability.
*Disclaimer: This is a prototype for academic evaluation only, not for clinical use.*
""")
st.divider()

# ==========================================
# 2. LOAD THE AI MODEL
# ==========================================
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, 'mura_efficientnet_10epochs.pth')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.warning(f"⚠️ Model weights not found at {model_path}. Please ensure the file is there.")
        
    model.eval()
    return model

model = load_model()

# ==========================================
# 3. GRAD-CAM IMPLEMENTATION
# ==========================================
class EfficientNetGradCAM:
    """Extracts gradients and activations from the final conv layer to build the heatmap."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Hook into the final convolutional layer of EfficientNet
        target_layer = self.model.features[-1]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class):
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        # Generate the heatmap weights
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam) # Discard negative pixels (noise)
        
        cam = cam.detach().numpy()
        
        # Normalize to [0, 1]
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

def overlay_heatmap(img, heatmap):
    """Blends the Grad-CAM heatmap with the original X-Ray."""
    # Resize heatmap to match the original image
    heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
    
    # Apply the Jet colormap (Deep red = high activation, Blue = low)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend the images together
    img_np = np.array(img)
    superimposed = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(superimposed)

# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ==========================================
# 5. DASHBOARD INTERFACE
# ==========================================
uploaded_file = st.file_uploader("Upload an X-Ray Image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Changed to 3 columns to fit the original, the heatmap, and the results
    col1, col2, col3 = st.columns([1, 1, 1])
    
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.subheader("1. Patient Radiograph")
        st.image(image, caption="Original Input", use_container_width=True)
    
    with col3:
        st.subheader("3. AI Analysis")
        analyze_button = st.button("Run Diagnostic Scan 🔍", type="primary", use_container_width=True)
        
    if analyze_button:
        with st.spinner('Analyzing osseous structures and generating XAI maps...'):
            
            # 1. Run Standard Inference
            input_tensor = process_image(image)
            # Enable gradients specifically for Grad-CAM
            with torch.set_grad_enabled(True): 
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
            prob_normal = probabilities[0].item() * 100
            prob_abnormal = probabilities[1].item() * 100
            predicted_class = 1 if prob_abnormal >= 50.0 else 0
            
            # 2. Generate Grad-CAM Heatmap
            grad_cam = EfficientNetGradCAM(model)
            heatmap = grad_cam.generate_heatmap(input_tensor, predicted_class)
            heatmap_img = overlay_heatmap(image, heatmap)
            
            # 3. Display the XAI Heatmap in the middle column
            with col2:
                st.subheader("2. Grad-CAM Interpretability")
                st.image(heatmap_img, caption="Red indicates high neural activation", use_container_width=True)
                
            # 4. Display Results in the right column
            with col3:
                if predicted_class == 1:
                    st.error("🚨 **TRIAGE ALERT: ABNORMAL**")
                    st.write(f"**Fracture / Pathology Probability:** {prob_abnormal:.2f}%")
                    st.progress(int(prob_abnormal))
                else:
                    st.success("✅ **STATUS: NORMAL**")
                    st.write(f"**Healthy Bone Probability:** {prob_normal:.2f}%")
                    st.progress(int(prob_normal))