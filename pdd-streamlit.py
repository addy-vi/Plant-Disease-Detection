import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import numpy as np


def prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) 
    predictions = model.predict(input_arr)
    return np.argmax(predictions) 


with st.sidebar:
    side = option_menu(
        "Option Menu",
        ["Home","Disease Detection"]
)

#Main Page
if(side=="Home"):
    st.header("PLANT DISEASE DETECTION SYSTEM")
    st.markdown("""
    Welcome to the Plant Disease Detection System!
    
    Our goal is to help you efficiently identify plant diseases. Simply upload an image of your plant, system will analyze it and detect any signs of disease. We can protect crops and promote healthier harvests!
   
     ##### How It Working:
    - **Upload Image:** Go to the **Disease Detection** page and upload an image of a plant.
    - **Analysis:** System will process the image using algorithms to identify diseases.
    - **Results:** View the results.

    ##### About?
    - **Accuracy:** System utilizes machine learning techniques for accurate disease detection.
    - **Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results few seconds, allowing for quick decision-making.
    """)

elif(side=="Disease Detection"):
    st.header("Disease Detection")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Predict")):
        st.write("Our Prediction")
        index = prediction(test_image)
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success(f"Model is Predicting it's a {class_name[index]}")
