import streamlit as st
import numpy as np
import joblib

from PIL import Image
import cv2
from rembg import remove, new_session

# ================================
# Load your trained model
# ================================
model = joblib.load("model.pkl")

# ================================
# Initialize rembg session
# ================================
session = new_session()

# ================================
# Adapted Feature Extraction
# ================================
def extract_features_from_np(image_np):
    """
    Extract features from an already-loaded RGB image (numpy array).
    """
    try:
        # Convert RGB to BGR for OpenCV
        bgr_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        mask = np.ones(bgr_img.shape[:2], dtype=np.uint8) * 255
        bgr_pixels_masked = bgr_img[mask > 0]
        rgb_pixels_masked = image_np[mask > 0]

        if rgb_pixels_masked.size == 0:
            return np.zeros((1, 16))

        mean_R = np.mean(rgb_pixels_masked[:, 0])
        mean_G = np.mean(rgb_pixels_masked[:, 1])
        mean_B = np.mean(rgb_pixels_masked[:, 2])
        sum_rgb = mean_R + mean_G + mean_B
        nr = mean_R / sum_rgb if sum_rgb != 0 else 0
        ng = mean_G / sum_rgb if sum_rgb != 0 else 0
        nb = mean_B / sum_rgb if sum_rgb != 0 else 0
        gmr = mean_G / mean_R if mean_R != 0 else 0
        gmb = mean_G / mean_B if mean_B != 0 else 0
        lab_pixels = cv2.cvtColor(bgr_pixels_masked.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        mean_L = np.mean(lab_pixels[:, 0])
        mean_a = np.mean(lab_pixels[:, 1])
        mean_b = np.mean(lab_pixels[:, 2])
        hsv_pixels = cv2.cvtColor(bgr_pixels_masked.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        mean_H = np.mean(hsv_pixels[:, 0])
        mean_S = np.mean(hsv_pixels[:, 1])
        ycbcr_pixels = cv2.cvtColor(bgr_pixels_masked.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        mean_Y = np.mean(ycbcr_pixels[:, 0])
        gdr_denominator = mean_G + mean_R
        gdr = (mean_G - mean_R) / gdr_denominator if gdr_denominator != 0 else 0
        vi_denominator = (2 * mean_G + mean_R + mean_B)
        vi = (2 * mean_G - mean_R - mean_B) / vi_denominator if vi_denominator != 0 else 0

        feature_array = np.array([
            mean_R, mean_G, mean_B,
            nr, ng, nb,
            gmr, gmb,
            mean_L, mean_a, mean_b,
            gdr, mean_H, mean_Y, mean_S, vi
        ]).reshape(1, -1)

        return feature_array

    except Exception as e:
        st.error(f"⚠️ Error extracting features: {e}")
        return np.zeros((1, 16))

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Rice Leaf SPAD Value Predictor", page_icon="🌾")

# App title with HSTU logo
st.markdown(
    """
    <h1 style='display: flex; align-items: center; gap: 10px;'>
        🌾 Rice Leaf SPAD Value Predictor
        <img src='https://www.hstu.ac.bd/assets/images/logo.png' width='60' style='margin-bottom:0px;'/>
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    Welcome to the **Rice Leaf SPAD Value Predictor**!  
    Upload a clear image of a rice leaf (preferably on a plain background).  
    The app will automatically remove the background and predict the SPAD value, which is an indicator of leaf chlorophyll content and plant health.
    """
)

st.markdown("""
### How to use:
1. **Upload or capture** a clear photo of a single rice leaf.
2. **Wait** for the app to process your image.
3. **Read your SPAD value and advice** for your crop.
""")

with st.expander("ℹ️ About SPAD Value in Rice"):
    st.write(
        """
        - **SPAD value** is a measure of chlorophyll content in plant leaves, widely used for assessing nitrogen status and health in rice crops.
        - **Typical SPAD value ranges for rice:**
            - **< 30:** Low chlorophyll, possible nitrogen deficiency. Leaves may appear pale or yellowish.
            - **30–40:** Moderate chlorophyll, generally healthy but may benefit from additional nitrogen.
            - **> 40:** High chlorophyll, healthy and well-nourished leaves. Leaves are deep green.
        - Monitoring SPAD values helps optimize fertilizer use and improve rice yield.
        """
    )

# --- Two icon buttons for upload/capture ---
col1, col2 = st.columns(2)
with col1:
    upload_clicked = st.button("📁 Upload Image")
with col2:
    capture_clicked = st.button("📷 Capture Image")

uploaded_file = None

if upload_clicked:
    uploaded_file = st.file_uploader(
        "Upload a rice leaf image (PNG, JPG, JPEG)", 
        type=["png", "jpg", "jpeg"],
        key="upload"
    )
elif capture_clicked:
    camera_image = st.camera_input("Capture a rice leaf image", key="capture")
    if camera_image is not None:
        uploaded_file = camera_image

if uploaded_file is not None:
    with st.spinner("Processing image and predicting SPAD value..."):
        input_bytes = uploaded_file.read()
        output_bytes = remove(input_bytes, session=session)
        from io import BytesIO
        img_no_bg = Image.open(BytesIO(output_bytes))
        image_np = np.array(img_no_bg)
        features = extract_features_from_np(image_np)
        predicted_spad = model.predict(features)[0]

    st.success(f"🌱 **Predicted SPAD Value:** `{predicted_spad:.2f}`")

    if predicted_spad < 30:
        st.error("❌ Low SPAD value: Your leaf may lack nitrogen. Consider fertilizer.")
    elif predicted_spad < 40:
        st.warning("⚠️ Moderate SPAD value: Leaf is okay, but could benefit from more nitrogen.")
    else:
        st.success("✅ High SPAD value: Your rice leaf is healthy!")

    with st.expander("Show uploaded image"):
        st.image(uploaded_file, caption="Original uploaded image", use_column_width=True)

    st.caption("For best results, use clear, well-lit images of single rice leaves.")
else:
    st.warning("Please upload or capture a rice leaf image to get a SPAD prediction.")


