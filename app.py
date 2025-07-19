import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2

# ✅ Set page configuration
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)

# ✅ Title
st.title("😷 Face Mask Detection App")
st.write("Upload an image and check if a face mask is detected.")

# ✅ Load model
@st.cache_resource
def load_face_mask_model():
    model = load_model("face_mask.h5")  # Make sure this model file exists
    return model

model = load_face_mask_model()

# ✅ File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ✅ Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha channel if present

    x = preprocess_input(img_array)
    x = np.expand_dims(x, axis=0)

    # ✅ Predict
    prediction = model.predict(x)[0][0]

    if prediction < 0.5:
        st.success("✅ **Mask Detected**")
    else:
        st.error("❌ **No Mask Detected**")
