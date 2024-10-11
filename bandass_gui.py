import nbformat
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
import plotly.express as px
from nbconvert import PythonExporter
import joblib
import pickle

################################
# Main app structure
st.set_page_config(page_title="Bandaas", layout="wide")

@st.cache_resource
def load_image(image_path):
    return Image.open(image_path)

################################


def page_image_display():
    st.title("LOOK OUT!!!")
    image = load_image("two_cars.png")
    st.image(image, caption="We are a collabortive Team!")

################################


def Page_overview():
    st.title("Overview")
    st.write(f"The link for the GitHub Repo")
    st.write(f"The link for the Documentation")
    st.write(f"The link for the Presentation")

################################

@st.cache_resource
def Page_about_data():
    st.title("About data")
    st.subheader("Let's dive deeper into the data!")
    st.write("## Data Description")
    st.write("This data is from the UK government's National Transportation Safety Board (NTSB). It provides details about road accidents in the UK.")
    st.write("## Data Columns")

    st.markdown("""
        * Accident_Index: Unique identifier for each accident
        * Location_Easting: Easting coordinate for the accident location 
        - Location_Northing: Northing coordinate for the accident location
        - Longitude: Longitude coordinate for the accident location
        - Latitude: Latitude coordinate for the accident location
        - Police_Force: Police force responsible for the accident
        - Accident_Severity: Severity of the accident (1:Fatal, 2: Serious, 3: Slight)
        - Number_of_Vehicles: Number of vehicles involved in the accident
        - Number_of_Casualties: Number of casualties involved in the accident
        - Date: Date of the accident
        - Day_of_Week: Day of the week of the accident
        - Time: Time of the accident
        - Local_Authority_(District): Local authority district where the accident occurred
        - Local_Authority_(Highway): Local authority highway where the accident occurred
        - 1st_Road_Class: Class of the first road where the accident occurred
        - 1st_Road_Number: Number of the first road where the accident occurred
        - Road_Type: Type of the road where the accident occurred
        - Speed_limit: Speed limit at the time of the accident
        - Junction_Detail: Detail of the junction where the accident occurred
        - Junction_Control: Control of the junction where the accident occurred
        - 2nd_Road_Class: Class of the second road where the accident occurred
        - 2nd_Road_Number: Number of the second road where the accident occurred
        - Pedestrian_Crossing-Human_Control: Human control of pedestrian crossing
        - Pedestrian_Crossing-Physical_Facilities: Physical facilities of pedestrian crossing
        - Light_Conditions: Lighting conditions at the time of the accident
        - Weather_Conditions: Weather conditions at the time of the accident
        - Road_Surface_Conditions: Road surface conditions at the time of the accident
        - Special_Conditions_at_Site: Special conditions at the site of the accident
        - Carriageway_Hazards: Carriageway hazards at the time of the accident
        - Urban_or_Rural_Area: Urban or rural area where the accident occurred
        - Did_Police_Officer_Attend_Scene_of_Accident: Did police officers attend the scene of the accident
        - LSOA_of_Accident_Location: Lower Super Output Area where the accident occurred
        - Vehicle_Reference: Vehicle details (e.g., make, model, year)
        - Casualty_Reference: Casualty details (e.g., make, model, year)
        - Casualty_Class: Casualty class (e.g., driver, passenger, pedestrian)
        - Sex_of_Casualty: Sex of the casualty
        - Age_of_Casualty: Age of the casualty
        - Age_Band_of_Casualty: Age band of the casualty
        - Casualty_Severity: Severity of the casualty (e.g., fatal, serious, slight)
        - Pedestrian_Location: Location of the pedestrian (e.g., sidewalk, crosswalk)
        - Pedestrian_Movement: Movement of the pedestrian (e.g., walking, standing, cycling)
        - Car_Passenger: Car passenger details (e.g., make, model, year)
        - Bus_or_Coach_Passenger: Bus or coach passenger details (e.g., make, model, year)
        - Pedestrian_Road_Maintenance_Worker: Pedestrian road maintenance worker details (e.g., make, model, year)
        
    """)

    st.write("## Acknowledgements")
    st.write("This dataset is provided by the UK government's National Transportation Safety Board (NTSB).")
    st.write("For more information, visit:https://www.kaggle.com/datasets/benoit72/uk-accidents-10-years-history-with-many-variables")
    st.write("## Data Preparation")
    st.write(
        "The data has been cleaned, filtered, and transformed to prepare it for analysis.")
    # ! the cleaned datase link
    st.write(r"For more information, visit: Link for the cleaned dataset")
    st.write("## Data Visualization")
    st.write(
        "We have created various visualizations to help you understand the data and gain insights.")
    st.write("## Data Analysis")
    st.write("We have performed various statistical analyses and visualizations to help you gain insights into the data.")
    st.write("## Data Export")
    st.write("We have provided various options to export the data in various formats.")
    st.write("## Data Contribution")
    st.write("We invite you to contribute to the data by providing feedback, reporting issues, or requesting additional data.")
    st.write("## Data Feedback")
    st.write(
        "If you have any questions, concerns, or feedback, please contact us at:  ")

################################


def page_model():
    st.title("Model")
    st.subheader("Let's create a predictive model!")
    st.write("## Model Description")
    st.write("We have developed a predictive model using machine learning algorithms.")
    st.write("## Model Architecture")
    st.write(
        "We have used a combination of linear regression, decision trees, and random forests.")
    st.write("## Model Performance")
    st.write(
        f"The model has achieved an accuracy of **99%** ISA :D  on a validation dataset.")
    st.write("## Model Evaluation")
    st.write("We have evaluated the model using various evaluation metrics, such as mean absolute error, mean squared error, and R-squared.")
    st.write("## Model Deployment")
    st.write("We have deployed the model as a web service using streamlit library.")
    st.write("## Model Contribution")
    st.write("We invite you to contribute to the model by improving its")

################################

@st.cache_resource
def page_authors():
    st.title("Teammates")
    st.write("## Names")
    st.write("""
    - Ahmed Yasser
    - Ahmed Abd El-Hameed
    - Abram Maher
    - Sarah Selim
    - Naglaa Reda
    """)
    st.write(" We wish you enjoy this journey!")

################################
# def save_feedback_to_excel(feedback):
#     file_path = "feedbacks.xlsx"

#     df = pd.DataFrame({"Feedback": [feedback]})

#     try:
#         book = load_workbook(file_path)
#         writer = pd.ExcelWriter(file_path, engine='openpyxl')
#         writer.book = book
#         writer.sheets = {ws.title: ws for ws in book.worksheets}
#         reader = pd.read_excel(file_path)
#         df.to_excel(writer, index=False, header=False, startrow=len(reader) + 1)
#     except FileNotFoundError:
#         df.to_excel(file_path, index=False)

#     writer.save()
#     writer.close()

################################
def Page_feedback():
    st.title("Your feedback")
    user_input = st.text_input("Enter your feedback or data")
    if st.button("Submit"):
        if user_input:
            # save_feedback_to_excel(user_input)
            st.write(f"Thank you for your feedback!")
        else:
            st.warning("Please enter your feedback before submitting!")

################################
################################
# def Demo():
#     st.warning("#@Stay Tuned@#")
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
# model = joblib.load("your_model_file.pkl")

@st.cache_resource
def load_model():
    # return joblib.load("your_model_file.pkl")
    pass

model = load_model()

def page_prediction():
    st.title("Prediction Section")

    # Dropdown for prediction choice
    prediction_type = st.selectbox("Select what you'd like to predict", [
                                   "Classification 1", "Classification 2", "Regression"])

    # Input fields for user to provide data
    st.write("Please provide the following details:")
    input1 = st.text_input("Input 1:")
    input2 = st.text_input("Input 2:")
    input3 = st.text_input("Input 3:")
    input4 = st.text_input("Input 4:")
    input5 = st.text_input("Input 5:")

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
        st.success(f"Prediction result: {prediction_type[0]}") # ! Need to be modified to Hameedoo



################################
MAIN_PAGES = [
    "Sample Data",
    "Emergency Assistant"]

SUB_PAGES = {
    "Hello": page_image_display,
    "Overview": Page_overview,
    "About Data": Page_about_data,
    "Model": page_model,
    "Team Names": page_authors,
    "Feedback": Page_feedback,

}

EM_SUBPAGES = {
    "Visuals": page_visuals,
    "Model output": page_prediction,
}
################################


def main():
    st.sidebar.title(f"Let's take a journey!")
    selection = st.sidebar.selectbox("Go to", MAIN_PAGES, index=0)
    if selection == "Sample Data":
        selection2 = st.sidebar.radio("Stage", list(SUB_PAGES.keys()))
        page = SUB_PAGES[selection2]
        page()
    elif selection == "Emergency Assistant":
        selection3 = st.sidebar.radio("Stage2", list(EM_SUBPAGES.keys()))
        page = EM_SUBPAGES[selection3]
        page()


if __name__ == "__main__":
    main()
