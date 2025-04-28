import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model_path = 'modelvgg.h5'
model = load_model(model_path)

# Define the input shape
input_shape = (224, 224)

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize(input_shape)
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit UI elements
st.title("Skin Cancer Classification")
st.write("This is a Streamlit app to classify skin cancer images as malignant or benign using a pre-trained model.")

# Upload image file
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
    
    # Load the image and preprocess
    img = Image.open(uploaded_image)
    img_preprocessed = preprocess_image(img)
    
    # Make predictions
    prediction = model.predict(img_preprocessed)
    
    # Interpret the prediction
    if prediction > 0.5:
        st.write("Prediction: Malignant")
    else:
        st.write("Prediction: Benign")

    # Show the prediction probability
    st.write(f"Prediction Probability: {prediction[0][0]:.2f}")

