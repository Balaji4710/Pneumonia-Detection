import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from main import get_severity
from database import init_db, save_to_db
st.set_page_config(page_title="Pneumonia Diagnosis through AI", layout="wide")
init_db()


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    try:
        model.load_state_dict(torch.load("pneumonia_model.pth", map_location="cpu"))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("🚨 'pneumonia_model.pth' not found!")
        return None


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
    st.title("🫁 Pneumonia AI Diagnostic Portal")
    st.markdown(
        "<span style='color:red'>Upload an X-ray to Detect Pneumonia or Normal. Disclaimer: !! Records are stored in the backend for training purposes.</span>",
        unsafe_allow_html=True
    )

    with st.sidebar:
        st.header("Patient Profile")
        p_name = st.text_input("Patient Name")
        p_age = st.number_input("Age", min_value=0, max_value=120, value=30)
        p_sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        p_weight = st.number_input("Weight", min_value=0.0, max_value=100.0, value=0.0 , step=0.1)
        p_height = st.number_input("Height", min_value=0.0, max_value=500.0, value=0.0 , step=0.01)
        p_work = st.text_input("Provide the Patient's Occupation ")
        st.divider()
        uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(image, use_container_width=True, caption="Current Scan")

        with col2:
            model = load_model()
            if model:
                label_idx, confidence, all_probs = predict(image, model)
                label = "PNEUMONIA" if label_idx == 1 else "NORMAL"
                sev_label, color = get_severity(all_probs[1])

                st.subheader("Diagnostic Report")
                if label == "PNEUMONIA":
                    st.error(f"Detection: {label}")
                else:
                    st.success(f"Detection: {label}")

                st.metric("Confidence Score", f"{confidence * 100:.2f}%")
                st.markdown(f"**Severity Level:** :{color}[{sev_label}]")


                if st.button("💾 Confirm & Archive Diagnosis"):
                    if p_name.strip():
                        save_to_db(p_name, p_age, p_sex, label, sev_label, confidence)
                        st.toast(f"Record for {p_name} successfully stored in backend database.", icon="✅")
                    else:
                        st.warning("⚠️ Enter patient name to save record.")


if __name__ == "__main__":
    main()