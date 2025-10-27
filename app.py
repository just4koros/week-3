# Save this as app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("MNIST Digit Classifier")

uploaded_file = st.file_uploader("Upload a digit image (28x28 grayscale)", type=["png", "jpg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L').resize((28,28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1,28,28,1)

    model = tf.keras.models.load_model('mnist_model.h5')
    prediction = model.predict(img_array)
    st.write(f"Predicted Digit: {np.argmax(prediction)}")
