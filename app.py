import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load my model from Colab
model = tf.keras.models.load_model("myfire_model.keras")

# Set title/icon and layout
st.set_page_config(page_title="IgnisAI", page_icon="🧿", layout="wide")
st.title("🌲🧿🔥 IgnisAI: Wildfire Detection from Satellite Image")


uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display mage
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess the image
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  

    # Predict
    prediction = model.predict(image_array)[0][0]  

    # Show probability
    if prediction > 0.5:
        st.info("Wildfire detected!")
    else:
        st.success("No wildfire detected.")
    
    # st.write(f"**Probability: {prediction * 100}%**")
    label = "🔥 Wildfire" if prediction >= 0.5 else "🌿 No Wildfire"
    confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100
    st.metric(label=label, value=f"{confidence:.2f}% Confidence")

