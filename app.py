import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

# Load trained model
model = load_model("mnist_model.h5")

# -----------------------------
# Custom Image Preprocessing Function
# -----------------------------
def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Invert image colors
    image = ImageOps.invert(image)

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert image to numpy array
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Reshape for model
    img_array = img_array.reshape(1, 28, 28)

    return img_array

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("MNIST Digit Prediction App")

st.write("Upload a handwritten digit image.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display uploaded image
    st.image(image, caption="Uploaded Image", width=200)

    # Preprocess image
    processed_image = preprocess_image(image)

    # Prediction
    prediction = model.predict(processed_image)

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"Predicted Digit: {predicted_digit}")

    st.info(f"Confidence: {confidence:.2f}%")