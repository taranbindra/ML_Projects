import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load the pre-trained model
model = tf.keras.models.load_model(r"C:\Users\Taranjot\Downloads\drowiness_model.h5")
labels_new = ["yawn", "no_yawn", "Closed", "Open"]
IMG_SIZE = 145


# Function to prepare the image for prediction
def prepare_image(image):
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize pixel values to range [0, 1]
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # Resize the image to 145x145
    return resized_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # Reshape to (1, 145, 145, 3)


# Streamlit interface
st.title("Driver Drowsiness Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Results :- ")

    prepared_image = prepare_image(image)

    # Make predictions
    predictions = model.predict(prepared_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = labels_new[predicted_class]

    if predicted_label in ["Open", "no_yawn"]:
        st.success("Driver is active.")
    elif predicted_label in ["Closed", "yawn"]:
        st.warning("Driver is sleepy.")
