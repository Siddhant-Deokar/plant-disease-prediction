import streamlit as st
from  PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json

with open('data.json', 'r') as file:
    data = json.load(file)  

model = load_model('models\my_model_1.keras')

class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus','Tomato___healthy']

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image):

    image = Image.open(image)

    img = image.resize((128, 128))
    
    # Convert the image to a numpy array
    img_array = np.array(img)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(image):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[int(predicted_class_index)]
    return predicted_class_index,predicted_class_name



st.title("Plant Disease Predictor")
    
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    image = image.resize((200,200))


    _, second_col,_ = st.columns(3) 
    
    second_col.image(image, caption="Uploaded Image")

    _,_,sample,_,_ = st.columns(5)

    if sample.button("Predict"):
        
        index, prediction = predict_image_class(uploaded_image)

        leaf_class = data['plant_classes'][index]['class']
        leaf_class = ' '.join(leaf_class.split('_'))

        leaf_info = data['plant_classes'][index]['information']
        leaf_pres = data['plant_classes'][index]['prescription']
        leaf_prec = data['plant_classes'][index]['precautions']
        

        # _,sample_2 , _ = st.columns(3)
        _,sample_2,_ = st.columns([1,3,1])

        sample_2.subheader(leaf_class)
        
        st.subheader("Information:")
        st.write(leaf_info)
        
        st.subheader("Prescription:")
        st.write(leaf_pres)

        st.subheader("Precautions")
        for i in leaf_prec:
            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;• {i}")

_, footer, _= st.columns([1,2,1])

footer.markdown(
    """
    <h2>Follow us at <a href="https://www.instagram.com/your_profile" target="_blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Instagram_icon.png" width="40">
    </a></h2>
    
    """,
    unsafe_allow_html=True
)

