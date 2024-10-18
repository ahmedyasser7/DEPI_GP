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

################################
# Main app structure
st.set_page_config(page_title="Emergency insights", layout="wide", page_icon=r"GUI/home_images/siren.png")
################################
# Emergency assistance sub-pages:
# Model information
# def load_models():
#     model_paths = [
#         r"Models/Accident_Severity_Model_new.pkl",
#         r"Models/Casualty_Severity_Model_new.pkl",
#         r"Models/Mapping_Model.pkl",
#         r"Models/No_Of_Casualities_Model.pkl"
#     ]

#     models = []
#     for path in model_paths:
#         try:
#             with open(path, "rb") as file:
#                 models.append(pickle.load(file))
#         except Exception as e:
#             st.error(f"Error loading model from {path}: {str(e)}")
#             return None
#     return tuple(models)

# models = load_models()
# if models:
#     model1, model2, model3, model4 = models
# else:
#     st.stop()

def page_prediction():
    st.title("Prediction Section")
    prediction_type = st.selectbox("Select what you'd like to predict", ["Accident Severity", "Causality severity", "Number of Casualties", "Mapping"])
    
    if prediction_type == "Accident Severity":
        st.write("Please provide the following details:")
        input1 = st.text_input("Longitude")
        input2 = st.text_input("Latitude")
        input3 = st.text_input("Number of Vehicles")
        input4 = st.text_input("Number of Casualties")
        input5 = st.text_input("Day of Week")
        input6 = st.text_input("Local Authority (District)")
        input7 = st.text_input("Urban or Rural Area")
        input8 = st.text_input("Speed limit")
        input9 = st.text_input("Light Conditions")
        input10 = st.text_input("Weather Conditions")
        input11 = st.text_input("Road Surface Conditions")
        input12 = st.text_input("Special Conditions at Site")
        
        Accident_Severity = pd.DataFrame({
            'Longitude': [input1], 'Latitude': [input2], 'Number_of_Vehicles': [input3], 'Number_of_Casualties': [input4], 
            'Day_of_Week': [input5], 'Local_Authority_(District)': [input6],'Urban_or_Rural_Area' :[input7],
            'Speed_limit': [input8], 'Light_Conditions': [input9], 'Weather_Conditions': [input10],
            'Road_Surface_Conditions': [input11], 'Special_Conditions_at_Site': [input12]
        }, index= [0])
        
        predict = st.button("Predict")
        if predict:
            with st.spinner("Predicting..."):
                if input1 and input2 and input3 and input4 and input5 and input6 and input7 and input8 and input9 and input10 and input11 and input12: 
                    # prediction = model1.predict(Accident_Severity)
                    # st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
                    pass
                else:
                    st.warning("Please fill all the inputs!")

    elif prediction_type == "Causality severity":
        st.write("Please provide the following details:")
        Casualty_type = {
            0: 'Pedestrian', 1: 'Cyclist', 2: 'Motorcycle 50cc and under rider or passenger',
            3: 'Motorcycle 125cc and under rider or passenger', 4: 'Motorcycle over 125cc and up to 500cc rider or passenger',
            5: 'Motorcycle over 500cc rider or passenger', 8: 'Taxi/Private hire car occupant', 9: 'Car occupant',
            10: 'Minibus (8 - 16 passenger seats) occupant', 11: 'Bus or coach occupant (17 or more pass seats)',
            16: 'Horse rider', 17: 'Agricultural vehicle occupant', 18: 'Tram occupant', 19: 'Van/Goods vehicle (3.5 tonnes mgw or under) occupant',
            20: 'Goods vehicle (over 3.5t. and under 7.5t.) occupant', 21: 'Goods vehicle (7.5 tonnes mgw and over) occupant', 
            22: 'Mobility scooter rider', 23: 'Electric motorcycle rider or passenger', 90: 'Other vehicle occupant',
            97: 'Motorcycle - unknown cc rider or passenger', 98: 'Goods vehicle (unknown weight) occupant'
        }
        
        input1 = st.text_input("Sex of Casualty")
        input2 = st.text_input("Age of Casualty")
        input3 = st.text_input("Car Passenger")
        input4 = st.text_input("Bus or Coach Passenger")
        input5 = st.selectbox("Choose Casualty Type", list(Casualty_type.values()))
        
        Casualty_Severity = pd.DataFrame({
            'Sex_of_Casualty': [input1], 'Age_of_Casualty': [input2], 'Car_Passenger': [input3], 'Bus_or_Coach_Passenger': [input4], 
            'Casualty_Type': [input5]
        }, index= [0])
        
        predict = st.button("Predict")
        if predict:
            with st.spinner("Predicting..."):
                if input1 and input2 and input3 and input4 and input5: 
                    # prediction = model2.predict(Casualty_Severity)
                    # st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
                    pass
                else:
                    st.warning("Please fill all the inputs!")

    elif prediction_type == "Number of Casualties":
        st.write("Please provide the following details:")
        input1 = st.text_input("Number of Vehicles")
        input2 = st.text_input("Speed limit")
        input3 = st.text_input("Light Conditions")
        input4 = st.text_input("Weather Conditions")
        input5 = st.text_input("Road Surface Conditions")
        
        Number_of_Casualties = pd.DataFrame({
            'Number_of_Vehicles': [input1], 'Speed_limit': [input2], 'Light_Conditions': [input3], 'Weather_Conditions': [input4], 
            'Road_Surface_Conditions': [input5]
        }, index= [0])
        
        predict = st.button("Predict")
        if predict:
            with st.spinner("Predicting..."):
                if input1 and input2 and input3 and input4 and input5: 
                    # prediction = model3.predict(Number_of_Casualties)
                    # st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
                    pass
                else:
                    st.warning("Please fill all the inputs!")

    elif prediction_type == "Mapping":
        st.write("Please provide the following details:")
        input1 = st.text_input("Was Vehicle Left Hand Drive?")
        input2 = st.text_input("Age of Driver")
        input3 = st.text_input("Age of Vehicle")
        
        Accident_involved = pd.DataFrame({
            'Was_Vehicle_Left_Hand_Drive': [input1], 'Age_of_Driver': [input2], 'Age_of_Vehicle': [input3]
        }, index= [0])
        
        predict = st.button("Predict")
        if predict:
            with st.spinner("Predicting..."):
                if input1 and input2 and input3: 
                    # prediction = model4.predict(Accident_involved)
                    # st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
                    pass
                else:
                    st.warning("Please fill all the inputs!")

################################
def main():
    page = page_prediction
    page()

if __name__ == "__main__":
    main()