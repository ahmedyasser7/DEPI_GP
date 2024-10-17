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
st.set_page_config(page_title="Emergency insights", layout="wide", page_icon= r"GUI\images\siren.png")
################################
# Emergency assitance sub-pages:

# def display_visual_from_notebook(notebook_path, cell_index):
#     with open(notebook_path, 'r', encoding='utf-8') as nb_file:
#         nb_content = nbformat.read(nb_file, as_version=4)

#         code = nb_content.cells[cell_index]['source']
#         exec(code, globals())


questions = [
    "How does the weather impact the number or severity of an accident?",
    "Does driver age have an effect on the number of accidents?",
    "What is the relation between hour, day, week, and month with several fatal accidents?",
    "Are certain car models safer than others?",
    "Is the social class of a casualty dependent on the accident severity?",
    "Can you forecast the future daily/weekly/monthly accidents?",
    "What about fatal accidents—can you predict them?",
    "Can you predict if an accident was fatal? (like Titanic prediction)"
]


def page_visuals():
    st.title("Visuals")
    selected_question = st.selectbox("Please, choose a question", questions)

    question_map = {
        questions[0]: 2,
        questions[1]: 3,
        questions[2]: 4,
        questions[3]: 5,
        questions[4]: 6,
        questions[5]: 7,
        questions[6]: 8,
        questions[7]: 9,
    }

    # notebook_path = "model.ipynb"
    cell_index = question_map[selected_question]

    # display_visual_from_notebook(notebook_path, cell_index)

################################
# Model information

# def load_models():
#     with open(r"Models\Accident_Severity_Model.pkl", "rb") as file1:
#         model1 = pickle.load(file1)
#     with open(r"Models\Casualty_Severity_Model.pkl", "rb") as file2:
#         model2 = pickle.load(file2)
#     with open(r"Models\Mapping_Model.pkl", "rb") as file3:
#         model3 = pickle.load(file3)
#     with open(r"Models\No_Of_Casualities_Model.pkl", "rb") as file4:
#         model4 = pickle.load(file4)
#     return model1, model2, model3, model4

# models = load_models()
# model1 = pickle.load(open(r"Models\Accident_Severity_Model.pkl", "rb"))
# model2 = pickle.load(open(r"Models\Casualty_Severity_Model.pkl", "rb"))
# model3 = pickle.load(open(r"Models\Mapping_Model.pkl", "rb"))
# model4 = pickle.load(open(r"Models\No_Of_Casualities_Model.pkl", "rb"))
@st.cache_resource
def load_models():
    model_paths = [
        r"Models\Accident_Severity_Model.pkl",
        r"Models\Casualty_Severity_Model_new.pkl",
        r"Models\Mapping_Model.pkl",
        r"Models\No_Of_Casualities_Model.pkl"
    ]

    models = []

    for path in model_paths:
        with open(path, "rb") as file:
            models.append(pickle.load(file))

    return tuple(models)

models = load_models()
model1, model2, model3, model4 = models
@st.cache_resource
def page_prediction():
    st.title("Prediction Section")
    prediction_type = st.selectbox("Select what you'd like to predict", [
                                   "Accident Severity", "Causuality severity", "Number of Casualities", "Mapping"])
    
    if prediction_type == "Accident Severity":
        st.write("Please provide the following details:")
        input1 = st.text_input("Longitude")
        input2 = st.text_input("Latitude")
        input3 = st.text_input("Number_of_Vehicles")
        input4 = st.text_input("Number_of_Casualties")
        input5 = st.text_input("Day_of_Week")
        input6 = st.text_input("Local_Authority_(District)")
        input7 = st.text_input("Urban_or_Rural_Area")
        input8 = st.text_input("Speed_limit")
        input9 = st.text_input("Light_Conditions")
        input10 = st.text_input("Weather_Conditions")
        input11 = st.text_input("Road_Surface_Conditions")
        input12 = st.text_input("Special_Conditions_at_Site")
        
        Accident_Severity = pd.DataFrame({
            'Longitude': [input1], 'Latitude': [input2], 'Number_of_Vehicles': [input3], 'Number_of_Casualties': [input4], 
            'Day_of_Week': [input5], 'Local_Authority_(District)': [input6],'Urban_or_Rural_Area' :[input7],
            'Speed_limit': [input8], 'Light_Conditions': [input9], 'Weather_Conditions': [input10],
            'Road_Surface_Conditions': [input11], 'Special_Conditions_at_Site': [input12]
        }, index= [0])
        
        predict = st.button("Predict")
        if predict:
            with st.spinner("Precicting..."):
                if input1 and input2 and input3 and input4 and input5 and input6 and input7 and input8 and input9 and input10 and input11 and input12: 
                    # Pass the input data to the model for prediction
                    prediction = model1.predict(Accident_Severity)
                    # Show prediction results to user
                    st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
                else:
                    st.warning("Please fill all the inputs!")
            st.success(f"Prediction result: {prediction_type[0]}")

    elif prediction_type == "Causuality severity":
        st.write("Please provide the following details:")
        Casualty_type = {0: 'Pedestrian', 1: 'Cyclist', 2: 'Motorcycle 50cc and under rider or passenger',
        3: 'Motorcycle 125cc and under rider or passenger', 4: 'Motorcycle over 125cc and up to 500cc rider or  passenger',
        5: 'Motorcycle over 500cc rider or passenger', 8: 'Taxi/Private hire car occupant', 
        9: 'Car occupant', 10: 'Minibus (8 - 16 passenger seats) occupant', 11: 'Bus or coach occupant (17 or more pass seats)', 
        16: 'Horse rider', 17: 'Agricultural vehicle occupant', 18: 'Tram occupant', 19: 'Van / Goods vehicle (3.5 tonnes mgw or under) occupant',
        20: 'Goods vehicle (over 3.5t. and under 7.5t.) occupant', 21: 'Goods vehicle (7.5 tonnes mgw and over) occupant', 
        22: 'Mobility scooter rider', 23: 'Electric motorcycle rider or passenger', 90: 'Other vehicle occupant', 
        97: 'Motorcycle - unknown cc rider or passenger', 98: 'Goods vehicle (unknown weight) occupant'}
        
        input1 = st.text_input("Sex_of_Casualty")
        input2 = st.text_input("Age_of_Casualty")
        input3 = st.text_input("Car_Passenger")
        input4 = st.text_input("Bus_or_Coach_Passenger")
        input5 = st.selectbox("Choose", Casualty_type.values())
        Casualty_Severity = pd.DataFrame({
            'Sex_of_Casualty': [input1], 'Age_of_Casualty': [input2], 'Car_Passenger': [input3], 'Bus_or_Coach_Passenger': [input4], 
            'Casualty_Type': [input5]
        }, index= [0])
        
        predict = st.button("Predict")
        if predict:
            with st.spinner("Precicting..."):
                if input1 and input2 and input3 and input4 and input5: 
                    # Pass the input data to the model for prediction
                    prediction = model2.predict(Casualty_Severity)
                    # Show prediction results to user
                    st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
                else:
                    st.warning("Please fill all the inputs!")
            st.success(f"Prediction result: {prediction_type[0]}")

    elif prediction_type == "Number of Casualities":
        st.write("Please provide the following details:")
        input1 = st.text_input("Number_of_Vehicles")
        input2 = st.text_input("Speed_limit")
        input3 = st.text_input("Light_Conditions")
        input4 = st.text_input("Weather_Conditions")
        input5 = st.text_input("Road_Surface_Conditions")
        #OUTPUT: Number_of_Casualties
        Number_of_Casualties = pd.DataFrame({
            'Number_of_Vehicles': [input1], 'Speed_limit': [input2], 'Light_Conditions': [input3], 'Weather_Conditions': [input4], 
            'Road_Surface_Conditions': [input5]
        }, index= [0])
        
        predict = st.button("Predict")
        if predict:
            with st.spinner("Precicting..."):
                if input1 and input2 and input3 and input4 and input5: 
                    # Pass the input data to the model for prediction
                    prediction = model3.predict(Number_of_Casualties)
                    # Show prediction results to user
                    st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
                else:
                    st.warning("Please fill all the inputs!")
            st.success(f"Prediction result: {prediction_type[0]}")
    elif prediction_type == "Mapping":
        st.write("Please provide the following details:")
        input1 = st.text_input("Was_Vehicle_Left_Hand_Drive?")
        input2 = st.text_input("Age_of_Driver")
        input3 = st.text_input("Age_of_Vehicle")
        # OUTPUT: Accident_involved
        Accident_involved = pd.DataFrame({
            'Was_Vehicle_Left_Hand_Drive': [input1], 'Age_of_Driver': [input2], 'Age_of_Vehicle': [input3]
        }, index= [0])
        
        predict = st.button("Predict")
        if predict:
            with st.spinner("Precicting..."):
                if input1 and input2 and input3: 
                    # Pass the input data to the model for prediction
                    prediction = model4.predict(Accident_involved)
                    # Show prediction results to user
                    st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
                else:
                    st.warning("Please fill all the inputs!")
            st.success(f"Prediction result: {prediction_type[0]}")

################################
EM_SUBPAGES = {
    "Visuals": page_visuals,
    "Model output": page_prediction,
}
################################
def main():
    selection = st.radio("", list(EM_SUBPAGES.keys()), index=0)
    page = EM_SUBPAGES[selection]
    page()

if __name__ == "__main__":
    main()