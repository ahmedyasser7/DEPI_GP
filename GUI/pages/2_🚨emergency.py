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
@st.cache_resource
def load_models():
    with open(r"Models\Accident_Severity_Model.pkl", "rb") as file1:
        model1 = pickle.load(file1)
    with open(r"Models\Casualty_Severity_Model.pkl", "rb") as file2:
        model2 = pickle.load(file2)
    with open(r"Models\Mapping_Model.pkl", "rb") as file3:
        model3 = pickle.load(file3)
    with open(r"Models\No_Of_Casualities_Model.pkl", "rb") as file4:
        model4 = pickle.load(file4)
    return model1, model2, model3, model4

# models = load_models() 

def page_prediction():
    st.title("Prediction Section")

    # Dropdown for prediction choice
    prediction_type = st.selectbox("Select what you'd like to predict", [
                                   "Accident Severity", "Causuality severity", "Number of Casualities", "Mapping"])
    
    if prediction_type == "Accident Severity":
        st.write("Please provide the following details:")
        input1 = st.text_input("Longitude:")
        input2 = st.text_input("Latitude")
        input3 = st.text_input("Number_of_Vehicles")
        input4 = st.text_input("Number_of_Casualties")
        input5 = st.text_input("Day_of_Week")
        input6 = st.text_input("Local_Authority_(District)")
        input7 = st.text_input("Local_Authority_(Highway)")
        input8 = st.text_input("Speed_limit")
        input9 = st.text_input("Light_Conditions")
        input10 = st.text_input("Weather_Conditions")
        input11 = st.text_input("Road_Surface_Conditions")
        input12 = st.text_input("Special_Conditions_at_Site")
        input13 = st.text_input("Urban_or_Rural_Area")
        # OUTPUT: Accident_Severity

    elif prediction_type == "Causuality severity":
        st.write("Please provide the following details:")
        input1 = st.text_input("Sex_of_Casualty")
        input2 = st.text_input("Age_of_Casualty")
        input3 = st.text_input("Car_Passenger")
        input4 = st.text_input("Bus_or_Coach_Passenger")
        input5 = st.text_input("Casualty_Type")
        #OUTPUT: Casualty_Severity

    elif prediction_type == "Number of Casualities":
        st.write("Please provide the following details:")
        input1 = st.text_input("Number_of_Vehicles")
        input2 = st.text_input("Speed_limit")
        input3 = st.text_input("Light_Conditions")
        input4 = st.text_input("Weather_Conditions")
        input5 = st.text_input("Road_Surface_Conditions")
        #OUTPUT: Number_of_Casualties
        
    elif prediction_type == "Mapping":
        st.write("Please provide the following details:")
        input1 = st.text_input("Was_Vehicle_Left_Hand_Drive?")
        input2 = st.text_input("Age_of_Driver")
        input3 = st.text_input("Age_of_Vehicle")
        # OUTPUT: Accident_involved

    # When user clicks the "Predict" button
    if st.button("Predict"):
        with st.spinner("Precicting..."):
            if input1 and input2 and input3 and input4 and input5:
                # Convert inputs into a format that your model expects (example: list or numpy array)
                input_data = np.array([[input1, input2, input3, input4, input5]])

                # Pass the input data to the model for prediction
                # prediction = model.predict(input_data)

                # Show prediction results to user
                # st.write(f"Prediction result for {prediction_type}: {prediction[0]}")
            else:
                st.warning("Please fill all the inputs!")
        st.success(f"Prediction result: {prediction_type[0]}") # ! Need to be modified by Hameedoo

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