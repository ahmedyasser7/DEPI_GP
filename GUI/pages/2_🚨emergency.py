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
    base_dir = r"D:\DEPI\Technical\DEPI_GP\GUI\pages"
    model_paths = [
        os.path.join(base_dir, "Accident_Severity_Model_new.pkl"),
        os.path.join(base_dir, "Casualty_Severity_Model_old.pkl"),
        os.path.join(base_dir, "Mapping_Model.pkl"),
        os.path.join(base_dir, "No_Of_Casualities_Model.pkl")
    ]

    models = []
    for path in model_paths:
        try:
            with open(path, "rb") as file:
                models.append(pickle.load(file))
        except Exception as e:
            st.error(f"Error loading model from {path}: {str(e)}")
            return None
    return tuple(models)


models = load_models()
if models:
    model1, model2, model3, model4 = models
else:
    st.stop()


def page_prediction():
    st.title("Prediction Section")
    prediction_type = st.selectbox("Select what you'd like to predict", [
                                   "Accident Severity", "Causality severity", "Number of Casualties", "Mapping"])

    if prediction_type == "Accident Severity":
        st.write("Please provide the following details:")
        days = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday',
                4: 'Wednesday', 5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
        light_conditions = {1: 'Daylight', 4: 'Darkness - lights lit', 5: 'Darkness - lights unlit',
                            6: 'Darkness - no lighting', 7: 'Darkness - lighting unknown'}
        weather = {1: 'Fine no high winds', 2: 'Raining no high winds', 3: 'Snowing no high winds', 4: 'Fine + high winds',
                   5: 'Raining + high winds', 6: 'Snowing + high winds', 7: 'Fog or mist', 8: 'Other', 9: 'Unknown'}
        road = {1: 'Dry', 2: 'Wet or damp', 3: 'Snow', 4: 'Frost or ice',
                5: 'Flood over 3cm. deep', 6: 'Oil or diesel', 7: 'Mud'}
        special = {0: 'nan', 1: 'Auto traffic signal - out', 2: 'Auto signal part defective',
                   3: 'Road sign or marking defective or obscured', 4: 'Roadworks', 5: 'Road surface defective', 6: 'Oil or diesel', 7: 'Mud'}
        rural = {0: 'Urban', 1: 'Rural', 2: 'Unallocated'}

        input1 = st.slider('Select a Longtiude', -4.8, 1.7, step=0.1)  # done
        input2 = st.slider('Select a Latitude', -2.00000,
                           6.00000, step=0.1)  # done
        input3 = st.slider('Select a number of vehicles',
                           0, 100, step=1)  # done
        input4 = st.slider('Select a number of casualties',
                           0, 100, step=1)  # done
        input5 = st.selectbox("Choose Day of the week",
                              list(days.values()))  # done
        input6 = st.slider(
            'Select a Local_Authority_(District)', -1.4, 2.3, step=0.1)  # done
        input7 = st.selectbox("Choose location", list(rural.values()))  # done
        input8 = st.slider('Select a speed limit', 0, 200, step=5)  # done
        input9 = st.selectbox("Choose the condition of the light", list(
            light_conditions.values()))  # done
        input10 = st.selectbox(
            # done
            "Choose the condition of the weather", list(weather.values()))
        input11 = st.selectbox(
            "Choose the condition of the Road", list(road.values()))  # done
        input12 = st.selectbox(
            "Choose the speacial condition", list(special.values()))  # done

        Accident_Severity_input = np.array(
            [input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12])
        predict = st.button("Predict")
        if predict:
            with st.spinner("Predicting..."):
                if input1 and input2 and input3 and input4 and input5 and input6 and input7 and input8 and input9 and input10 and input11 and input12:
                    prediction = model1.predict(Accident_Severity_input)
                    st.write(f"Prediction result for {
                             prediction_type}: {prediction[0]}")
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
        gender = {0: 'Male', 1: 'Female'}
        passenger = {0: 'Not car passenger', 1: 'Front seat passenger', 2:	'Rear seat passenger'}
        bus = {0: 'Not a bus or coach passenger', 1: 'Boarding', 2: 'Alighting', 
               3: 'Standing passenger', 4: 'Seated passenger'}
        
        input1 = st.selectbox("Choose the gender",
                              list(gender.values()))  # done
        input2 = st.slider('Age of Casualty',
                           1, 100, step=1)  # done
        input3 = st.selectbox("Car Passenger",
                              list(passenger.values()))  # done
        input4 = st.selectbox("Bus or Coach Passenger",
                              list(bus.values())) # done
        input5 = st.selectbox("Choose Casualty Type",
                              list(Casualty_type.values()))  # done
        
        Casualty_Severity_input = np.array([input1, input2, input3, input4, input5])

        predict = st.button("Predict")
        if predict:
            with st.spinner("Predicting..."):
                if input1 and input2 and input3 and input4 and input5:
                    prediction = model2.predict(Casualty_Severity_input)
                    st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
                    # pass
                else:
                    st.warning("Please fill all the inputs!")

    elif prediction_type == "Number of Casualties":
        st.write("Please provide the following details:")
        road = {1: 'Dry', 2: 'Wet or damp', 3: 'Snow', 4: 'Frost or ice',
                5: 'Flood over 3cm. deep', 6: 'Oil or diesel', 7: 'Mud'}
        weather = {1: 'Fine no high winds', 2: 'Raining no high winds', 3: 'Snowing no high winds', 4: 'Fine + high winds',
                   5: 'Raining + high winds', 6: 'Snowing + high winds', 7: 'Fog or mist', 8: 'Other', 9: 'Unknown'}
        light_conditions = {1: 'Daylight', 4: 'Darkness - lights lit', 5: 'Darkness - lights unlit',
                            6: 'Darkness - no lighting', 7: 'Darkness - lighting unknown'}
        
        input1 = st.slider('Select a number of vehicles',
                           0, 100, step=1)  # done
        input2 = st.slider('Select a speed limit', 0, 200, step=5)  # done
        input3 = st.selectbox("Choose the condition of the light", list(
            light_conditions.values()))  # done
        input4 = st.selectbox("Choose the condition of the weather", list(weather.values())) # done
        input5 = st.selectbox(
            "Choose the condition of the Road", list(road.values()))  # done
        Number_of_Casualties_input = np.array([input1, input2, input3, input4, input5])
        

        predict = st.button("Predict")
        if predict:
            with st.spinner("Predicting..."):
                if input1 and input2 and input3 and input4 and input5:
                    prediction = model3.predict(Number_of_Casualties_input)
                    st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
                    # pass
                else:
                    st.warning("Please fill all the inputs!")

    elif prediction_type == "Mapping":
        st.write("Please provide the following details:")
        
        hand = {'No': 1, 'Yes': 2}
        
        input1 = st.selectbox("Was Vehicle Left Hand Drive?", list(
            hand.keys()))  # done
        input2 = st.slider("Age of Driver", 10, 100, step=1)  # done
        input3 = st.slider("Age of Vehicle", 10, 100, step=1)  # done
        
        Accident_involved_input = np.array([hand[input1], input2, input3]) 

        predict = st.button("Predict")
        if predict:
            with st.spinner("Predicting..."):
                if input1 and input2 and input3:
                    prediction = model4.predict(Accident_involved_input)
                    st.write(f"Prediction result for {
                             prediction_type}: {prediction[0]}")
                    # pass
                else:
                    st.warning("Please fill all the inputs!")

################################


def main():
    page = page_prediction
    page()


if __name__ == "__main__":
    main()
