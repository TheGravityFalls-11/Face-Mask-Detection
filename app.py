import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# Load the trained model
model = load_model("face_mask_detector.h5")

# These classes should match your training labels
labels = ["with_mask", "without_mask", "incorrect_mask"]  # Update as per your dataset

IMG_SIZE = 224

st.title("😷 Face Mask Detection App")
st.write("Upload an image and check if people are wearing masks.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_copy = image_np.copy()

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert to OpenCV format
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.write("No faces detected.")
    else:
        for (x, y, w, h) in faces:
            face = image_copy[y:y+h, x:x+w]
            try:
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            except Exception as e:
                st.warning(f"Skipping a face due to resize error: {e}")
                continue

            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face)[0]
            if len(pred) != len(labels):
                st.warning(f"Prediction length {len(pred)} does not match labels length {len(labels)}. Skipping this face.")
                continue

            label_idx = np.argmax(pred)
            label = labels[label_idx]
            confidence = pred[label_idx] * 100

            # Draw bounding box and label
            color = (0, 255, 0) if label == "with_mask" else (0, 0, 255) if label == "without_mask" else (0, 255, 255)
            label_text = f"{label} ({confidence:.2f}%)"
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image_copy, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

        st.image(image_copy, caption="Result", use_container_width=True)
