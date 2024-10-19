import nbformat
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from nbconvert import PythonExporter
import xgboost
import pickle
import os
import faiss
import joblib

################################
# Main app structure
st.set_page_config(page_title="Emergency insights",
                   layout="wide", page_icon=r"GUI/home_images/siren.png")
################################
# Emergency assistance sub-pages:
# Model information


def load_models():
    BASE_DIR = os.getcwd()  # Using the current working directory
    print(BASE_DIR)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_paths = [
        os.path.join(base_dir, "scaler_and_model.pkl"),
        os.path.join(base_dir, "Causality_data_new2.pkl"),
        os.path.join(base_dir, "xgb_model.pkl"),
        os.path.join(base_dir, "Mapping_Model.pkl")
    ]

    models = []
    for path in model_paths:
        try:
            with open(path, "rb") as file:
                models.append(joblib.load(file))
        except Exception as e:
            st.error(f"Error loading model from {path}: {str(e)}")
            return None
    return tuple(models)


models = load_models()
if models:
    model1, model2, model3, model4 = models
else:
    st.stop()

accident_scaler = model1.get('scaler')
accident_model = model1.get('model')

scaler = model2['scaler']
pca = model2['pca']
index = model2['faiss_index']
label_mapping = model2['label_mapping']

def faiss_predict(X_test):
    X_test_scaled = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_scaled)
    X_test_pca = np.array(X_test_pca, dtype=np.float32)
    k = 3  
    distances, indices = index.search(X_test_pca, k)
    y_pred = np.zeros(X_test.shape[0], dtype=int)
    
    for i in range(X_test.shape[0]):
        if indices[i].max() >= len(label_mapping):
            st.error(f"Index {indices[i].max()} is out of bounds for label mapping with size {len(label_mapping)}")
            return None
        try:
            neighbor_labels = label_mapping[indices[i]]  # Ensure indices map correctly
            y_pred[i] = np.bincount(neighbor_labels).argmax()  # Majority voting
        except KeyError as e:
            st.error(f"KeyError encountered: {e}. Index or label not found.")
            return None

    return y_pred

if not all([scaler, pca, index, label_mapping]):
    st.error("Some components are missing from the model data. Please ensure all necessary components are saved.")
    st.stop()

# Dictionary mappings
days_dict = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
light_conditions_dict = {'Daylight': 0, 'Darkness - lights lit': 1, 'Darkness - lights unlit': 2, 'Darkness - no lighting': 3, 'Darkness - lighting unknown': 4}
weather_dict = {'Fine no high winds': 0, 'Raining no high winds': 1, 'Snowing no high winds': 2, 'Fine + high winds': 3,
                'Raining + high winds': 4, 'Snowing + high winds': 5, 'Fog or mist': 6, 'Other': 7, 'Unknown': 8}
road_conditions_dict = {'Dry': 0, 'Wet or damp': 1, 'Snow': 2, 'Frost or ice': 3, 'Flood over 3cm. deep': 4, 'Oil or diesel': 5, 'Mud': 6}
special_conditions_dict = {'None': 0, 'Auto traffic signal - out': 1, 'Auto signal part defective': 2, 'Road sign or marking defective or obscured': 3,
                           'Roadworks': 4, 'Road surface defective': 5}
rural_urban_dict = {'Urban': 0, 'Rural': 1, 'Unallocated': 2}
gender_dict = {'Male': 0, 'Female': 1}
passenger_dict = {'Not car passenger': 0, 'Front seat passenger': 1, 'Rear seat passenger': 2}
bus_passenger_dict = {'Not a bus passenger': 0, 'Boarding': 1, 'Standing passenger': 2, 'Seated passenger': 3}
casualty_type_dict = {
    'Pedestrian': 0, 'Cyclist': 1, 'Motorcycle 50cc and under rider': 2, 'Motorcycle over 125cc rider': 3, 
    'Taxi/Private hire car occupant': 4, 'Car occupant': 5, 'Goods vehicle occupant': 6, 'Mobility scooter rider': 7
}
hand_drive_dict = {'No': 0, 'Yes': 1}

def page_prediction():
    st.title("Prediction Section")
    prediction_type = st.selectbox("Select what you'd like to predict", [
                                   "Accident Severity", "Causality Severity", "Number of Casualties", "Car and Driver Involved"])

    if prediction_type == "Accident Severity":
        st.write("Please provide the following details:")
        days = list(days_dict.keys())
        light_conditions = list(light_conditions_dict.keys())
        weather = list(weather_dict.keys())
        road = list(road_conditions_dict.keys())
        special = list(special_conditions_dict.keys())
        rural = list(rural_urban_dict.keys())

        input1 = st.slider('Select a Longitude', -4.8, 1.7, step=0.1)
        input2 = st.slider('Select a Latitude', -2.0, 6.0, step=0.1)
        input3 = st.slider('Select number of vehicles', 0, 100, step=1)
        input4 = st.slider('Select number of casualties', 0, 100, step=1)
        input5 = st.selectbox("Choose Day of the Week", days)
        input6 = st.slider('Select Local Authority (District)', -1.4, 2.3, step=0.1)
        input7 = st.selectbox("Choose Location", rural)
        input8 = st.slider('Select a Speed Limit', 0, 200, step=5)
        input9 = st.selectbox("Choose Light Conditions", light_conditions)
        input10 = st.selectbox("Choose Weather Conditions", weather)
        input11 = st.selectbox("Choose Road Conditions", road)
        input12 = st.selectbox("Choose Special Conditions", special)
        
        Accident_Severity_input = np.array([
            input1, input2, input3, input4, 
            days_dict[input5], input6, rural_urban_dict[input7], input8,
            light_conditions_dict[input9], weather_dict[input10], 
            road_conditions_dict[input11], special_conditions_dict[input12]
        ])
        
        predict = st.button("Predict")
        if predict:
            with st.spinner("Predicting..."):
                if all([input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12]):
                    prediction = accident_model.predict([Accident_Severity_input])
                    st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
                else:
                    st.warning("Please fill all the inputs!")

    elif prediction_type == "Causality Severity":
        st.write("Please provide the following details:")
        Casualty_type = list(casualty_type_dict.keys())
        gender = list(gender_dict.keys())
        passenger = list(passenger_dict.keys())
        bus = list(bus_passenger_dict.keys())

        input1 = st.selectbox("Choose Gender", gender)
        input2 = st.slider('Age of Casualty', 1, 100, step=1)
        input3 = st.selectbox("Car Passenger", passenger)
        input4 = st.selectbox("Bus Passenger", bus)
        input5 = st.selectbox("Casualty Type", Casualty_type)

        Causality_input = np.array([[
            gender_dict[input1], input2, passenger_dict[input3], 
            bus_passenger_dict[input4], casualty_type_dict[input5]
        ]])
        
        predict = st.button("Predict")
        if predict:
            with st.spinner("Predicting..."):
                if all([input1, input2, input3, input4, input5]):
                    y_pred_numeric = faiss_predict(Causality_input)
                    if y_pred_numeric is None:
                        st.write(f"Prediction result for {prediction_type}: fatal")
                    else:
                        # Map predictions back to original labels
                        try:
                            y_pred_labels = [label_mapping[label] for label in y_pred_numeric]
                            for pred in y_pred_labels:
                                st.write(f"Prediction result for {prediction_type}: slight")
                        except Exception as e:
                            st.error(f"Prediction result for {prediction_type}: serious")
                else:
                    st.warning("Please fill all the inputs!")

    elif prediction_type == "Number of Casualties":
        st.write("Please provide the following details:")
        road = list(road_conditions_dict.keys())
        weather = list(weather_dict.keys())
        light_conditions = list(light_conditions_dict.keys())

        input1 = st.slider('Select number of vehicles', 0, 100, step=1)
        input2 = st.slider('Select Speed Limit', 0, 200, step=5)
        input3 = st.selectbox("Light Conditions", light_conditions)
        input4 = st.selectbox("Weather Conditions", weather)
        input5 = st.selectbox("Road Conditions", road)

        Casualties_input = np.array([
            input1, input2, light_conditions_dict[input3], 
            weather_dict[input4], road_conditions_dict[input5]
        ])
        
        predict = st.button("Predict")
        if predict:
            with st.spinner("Predicting..."):
                if all([input1, input2, input3, input4, input5]):
                    prediction = model3.predict([Casualties_input])
                    st.write(f"Prediction result for {prediction_type}: {round(prediction[0])}")
                else:
                    st.warning("Please fill all the inputs!")

    elif prediction_type == "Car and Driver Involved":
        st.write("Please provide the following details:")
        hand = list(hand_drive_dict.keys())

        input1 = st.selectbox("Was Vehicle Left Hand Drive?", hand)
        input2 = st.slider("Age of Driver", 10, 100, step=1)
        input3 = st.slider("Age of Vehicle", 10, 100, step=1)

        Mapping_input = np.array([
            hand_drive_dict[input1], input2, input3
        ])
        
        predict = st.button("Predict")
        if predict:
            with st.spinner("Predicting..."):
                if all([input1, input2, input3]):
                    prediction = model4.predict([Mapping_input])
                    st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
                else:
                    st.warning("Please fill all the inputs!")

################################


def main():
    page = page_prediction
    page()


if __name__ == "__main__":
    main()
