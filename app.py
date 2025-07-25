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

        # Create a full mask (since background is already removed)
        mask = np.ones(bgr_img.shape[:2], dtype=np.uint8) * 255

        # Masked pixels (in BGR)
        bgr_pixels_masked = bgr_img[mask > 0]
        rgb_pixels_masked = image_np[mask > 0]

        if rgb_pixels_masked.size == 0:
            return {
                'R':0,'G':0,'B':0,
                'nr':0,'ng':0,'nb':0,
                'mgmr':0,'gmb':0,
                'L':0,'a':0,'b':0,
                'gdr':0,'H_mean':0,'Y_mean':0,'S':0,'VI':0
            }

        # Mean RGB
        mean_R = np.mean(rgb_pixels_masked[:, 0])
        mean_G = np.mean(rgb_pixels_masked[:, 1])
        mean_B = np.mean(rgb_pixels_masked[:, 2])

        # Normalized RGB
        sum_rgb = mean_R + mean_G + mean_B
        nr = mean_R / sum_rgb if sum_rgb != 0 else 0
        ng = mean_G / sum_rgb if sum_rgb != 0 else 0
        nb = mean_B / sum_rgb if sum_rgb != 0 else 0

        # Ratios
        mgmr = mean_G / mean_R if mean_R != 0 else 0
        gmb = mean_G / mean_B if mean_B != 0 else 0

        # LAB color space
        lab_pixels = cv2.cvtColor(bgr_pixels_masked.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        mean_L = np.mean(lab_pixels[:, 0])
        mean_a = np.mean(lab_pixels[:, 1])
        mean_b = np.mean(lab_pixels[:, 2])

        # HSV color space
        hsv_pixels = cv2.cvtColor(bgr_pixels_masked.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        mean_H = np.mean(hsv_pixels[:, 0])
        mean_S = np.mean(hsv_pixels[:, 1])

        # YCbCr luminance
        ycbcr_pixels = cv2.cvtColor(bgr_pixels_masked.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
        mean_Y = np.mean(ycbcr_pixels[:, 0])

        # Green-red difference ratio
        gdr_denominator = mean_G + mean_R
        gdr = (mean_G - mean_R) / gdr_denominator if gdr_denominator != 0 else 0

        # Vegetation Index
        vi_denominator = (2 * mean_G + mean_R + mean_B)
        vi = (2 * mean_G - mean_R - mean_B) / vi_denominator if vi_denominator != 0 else 0

        # Order must match model training order
        feature_array = np.array([
            mean_R, mean_G, mean_B,
            nr, ng, nb,
            mgmr, gmb,
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
st.title("🌿 SPAD Prediction Web App")
st.write("Upload a leaf image, background will be removed, and SPAD value predicted.")

uploaded_file = st.file_uploader("Upload a leaf image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_bytes = uploaded_file.read()
    st.write(f"✅ File read successfully: {len(input_bytes)} bytes")
    output_bytes = remove(input_bytes, session=session)
    # Then open the processed image
    from io import BytesIO
    from PIL import Image
    img_no_bg = Image.open(BytesIO(output_bytes))
    st.image(img_no_bg, caption="Background removed", use_column_width=True)

    # Convert to numpy (RGB)
    image_np = np.array(img_no_bg)


    # Extract features
    features = extract_features_from_np(image_np)

    # Predict SPAD
    predicted_spad = model.predict(features)[0]
    st.success(f"✅ Predicted SPAD Value: **{predicted_spad:.2f}**")
