import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
# Import the shared logic from your main script
from main import get_severity

# 1. Page Configuration
st.set_page_config(page_title="Pneumonia AI Diagnostic", page_icon="🧪", layout="wide")

# 2. UI Styling (Fixes white background bug)
st.markdown("""
    <style>
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05); 
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    try:
        model.load_state_dict(torch.load("pneumonia_model.pth", map_location="cpu"))
        model.eval()
    except FileNotFoundError:
        st.error("🚨 'pneumonia_model.pth' not found! Run main.py first.")
    return model


def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, pred = torch.max(probs, 1)
    return pred.item(), conf.item(), probs[0].tolist()


def main():
    st.title("🫁 Chest X-Ray Diagnostic Portal")
    st.write("Upload a digital X-ray for AI-assisted pneumonia screening.")
    st.divider()

    with st.sidebar:
        st.header("Upload Data")
        uploaded_file = st.file_uploader("Select X-Ray Image", type=["jpg", "jpeg", "png"])
        if st.button("Reset App"):
            st.rerun()

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.subheader("Inspection View")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("AI Analysis Report")
            model = load_model()
            if model:
                label_idx, conf, all_probs = predict(image, model)
                label = "PNEUMONIA" if label_idx == 1 else "NORMAL"

                # Use the imported severity logic
                pneumonia_prob = all_probs[1]
                severity_label, color = get_severity(pneumonia_prob)

                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    if label == "PNEUMONIA":
                        st.error(f"**Result:** {label}")
                    else:
                        st.success(f"**Result:** {label}")
                with res_col2:
                    st.metric("Infection Severity", severity_label)

                st.markdown(f"**Clinical Status:** :{color}[{severity_label}] ({conf * 100:.1f}%)")
                st.bar_chart({"Probability": all_probs})

                st.divider()
                if st.button("Confirm Diagnosis & Log"):
                    st.toast(f"Logged: {severity_label}", icon="✅")
    else:
        st.info("👋 Please upload a chest X-ray in the sidebar.")


if __name__ == "__main__":
    main()