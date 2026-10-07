import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Signature Identification System",
    page_icon="✍️",
    layout="centered"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("signature_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# =========================================================
# FUNCTIONS
# =========================================================

def preprocess_image(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        (128, 64)
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return gray


def extract_features(image):

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features.reshape(1, -1)


# =========================================================
# HEADER
# =========================================================

st.title("✍️ Signature Identification System")

st.markdown(
    "### Machine Learning Mini Project"
)

st.write(
    "This application identifies the student associated "
    "with a signature using HOG feature extraction and "
    "an SVM classifier."
)

st.divider()


# =========================================================
# PROJECT INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Students",
        "5"
    )

with col2:
    st.metric(
        "Original Images",
        "95"
    )

with col3:
    st.metric(
        "Test Accuracy",
        "73.68%"
    )


st.divider()


# =========================================================
# UPLOAD
# =========================================================

st.subheader("📤 Upload Signature")

uploaded_file = st.file_uploader(
    "Choose a signature image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(
            uploaded_file.read()
        ),
        dtype=np.uint8
    )

    image = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if image is not None:

        st.image(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            ),
            caption="Uploaded Signature",
            width="stretch"
        )

        st.write("")

        if st.button(
            "🔍 Identify Signature",
            type="primary",
            width="stretch"
        ):

            # Preprocessing
            processed_image = preprocess_image(
                image
            )

            # HOG feature extraction
            features = extract_features(
                processed_image
            )

            # Prediction
            prediction = model.predict(
                features
            )

            student_name = (
                label_encoder
                .inverse_transform(prediction)[0]
            )

            # Confidence
            probabilities = (
                model.predict_proba(features)[0]
            )

            confidence = (
                np.max(probabilities) * 100
            )

            st.divider()

            st.subheader(
                "🎯 Identification Result"
            )

            st.success(
                f"Predicted Student: **{student_name}**"
            )

            st.progress(
                int(confidence)
            )

            st.info(
                f"Model Confidence: **{confidence:.2f}%**"
            )


# =========================================================
# HOW IT WORKS
# =========================================================

st.divider()

st.subheader("⚙️ How the System Works")

st.markdown(
    """
**1️⃣ Upload Signature**  
The user uploads a signature image.

**2️⃣ Preprocessing**  
The image is converted to grayscale, resized and denoised.

**3️⃣ HOG Feature Extraction**  
Histogram of Oriented Gradients extracts shape and edge features.

**4️⃣ SVM Classification**  
The SVM model compares the extracted features with learned signature patterns.

**5️⃣ Identification**  
The system predicts the most likely student.
"""
)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.divider()

st.subheader("🧠 Machine Learning Model")

st.write(
    "**Feature Extraction:** HOG "
    "(Histogram of Oriented Gradients)"
)

st.write(
    "**Classifier:** Support Vector Machine (SVM)"
)

st.write(
    "**Kernel:** RBF"
)

st.write(
    "**Training Images:** 76"
)

st.write(
    "**Testing Images:** 19"
)

st.write(
    "**Test Accuracy:** 73.68%"
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Signature Identification System | "
    "Machine Learning Mini Project"
)