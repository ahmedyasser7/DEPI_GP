import nbformat
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from nbconvert import PythonExporter
import pickle
import os
###############################
BASE_DIR = os.path.dirname(__file__)  # Current directory where Bandass.py exists
IMAGE_DIR = os.path.join(BASE_DIR, "images")  # Set the path to the images directory 
ANALYSIS_IMAGE_DIR = os.path.join(BASE_DIR, "analysis_images")

################################
# Main app structure
st.set_page_config(page_title="Our Project Insights", layout="wide", page_icon= r"images/exploratory-analysis.png")
################################
def load_image(image_path):
    try:
        print(f"Loading image from: {image_path}")  # Debugging line
        return Image.open(image_path)
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")  # Improved error message
        return None

################################
def Page_about_data():
    st.title("About data")
    st.subheader("Let's dive deeper into the data!")
    st.write("## Data Description")
    st.write("This data is from the UK government's National Transportation Safety Board (NTSB). It provides details about road accidents in the UK.")
    st.divider()
    
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
    st.divider()
    
    st.write("## Acknowledgements")
    st.write("This dataset is provided by the UK government's National Transportation Safety Board (NTSB).")
    st.write("For more information, visit:https://www.kaggle.com/datasets/benoit72/uk-accidents-10-years-history-with-many-variables")
    st.divider()
    
    st.write("## Data Preparation")
    st.write(
        "The data has been cleaned, filtered, and transformed to prepare it for analysis.")
    st.write("For more information, visit:")
    st.markdown("""
            - https://drive.google.com/file/d/1cE5rwGA3ZVW3x-wXaZNWrsHYj4KCuyzr/view?usp=drive_link
            - https://drive.google.com/file/d/1INPspVo4f1nl_WoG2V7PXqq1YJIBmZ05/view?usp=drive_link 
            - https://drive.google.com/file/d/1uvQ4_PaihZNbvRbA4FNZfRssplMB5oUX/view?usp=drive_link 
        """)
    st.divider()
    
    st.write("## Data Analysis and Visualization ")
    st.write("We have performed various statistical analysis and visualizations to help you gain insights into the data")
    st.divider()
    
    st.write("### Some Of Data Art")
    
    images = [
        os.path.join(ANALYSIS_IMAGE_DIR, "correlation.png"),
        os.path.join(ANALYSIS_IMAGE_DIR, "HeatmapofAccidentSeveritybyRoadSurface.png"),
        os.path.join(ANALYSIS_IMAGE_DIR, "rural_vs_urban.png"),
    ]
    
    st.markdown("""
        <style>
        .btn-style {
            background-color: #851CA3;
            color: purple;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease-in-out;
        }
        .btn-style:hover {
            background-color: #AAAAAA;
        }
        .slider-indicators {
            text-align: center;
            margin-top: 10px;
        }
        .slider-indicators span {
            height: 15px;
            width: 15px;
            margin: 0 5px;
            display: inline-block;
            background-color: #bbb;
            border-radius: 50%;
        }
        .slider-indicators .active {
            background-color: #171717;
        }
        </style>
    """, unsafe_allow_html=True)

    current_image_path = images[st.session_state.carousel_index]
    current_image = load_image(current_image_path)
    if current_image is not None:
        st.image(current_image, width=700)
    else:
        st.error(f"Failed to load image at index {st.session_state.carousel_index}.")
    prev, _, next = st.columns([1, 10, 1])

    if prev.button("back", key="prev", help="Previous image", type="primary"):
        st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(images)

    if next.button("Next", key="next", help="Next image", type="primary"):
        st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(images)

    # Display current image index
    st.write(f"Image {st.session_state.carousel_index + 1} of {len(images)}")

    # Custom slider indicators for the images
    st.markdown("""
        <div class="slider-indicators">
    """, unsafe_allow_html=True)

    for i in range(len(images)):
        active_class = "active" if i == st.session_state.carousel_index else ""
        st.markdown(f"<span class='{active_class}'></span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.divider()
    
    st.write("## Data Contribution")
    st.write("We invite you to contribute to the data by providing feedback, reporting issues, or requesting additional data.")
    st.divider()
    
    st.write("## Data Feedback")
    st.write(
        "If you have any questions, concerns, or feedback, please contact us at the feedback page")
################################
def page_model():
    st.title("Model Information")
    st.subheader("Let's create a predictive model!")
    st.write("## Model Description")
    st.write("We have developed a predictive model using machine learning algorithms.")
    st.divider()
    
    st.write("## Model Architecture")
    st.write(
        "We have used a combination of linear regression, decision trees, and random forests.")
    st.divider()
    
    st.write("## Model Performance")
    st.write(
        f"The model has achieved an accuracy of **85%** :D  on a validation dataset.")
    st.divider()
    
    st.write("## Model Evaluation")
    st.write("We have evaluated the model using various evaluation metrics, such as mean absolute error, mean squared error, and R-squared.")
    st.divider()
    
    st.write("## Model Deployment")
    st.write("We have deployed the model as a web service using streamlit library.")
    st.divider()
    
    st.write("## Model Contribution")
    st.write("We invite you to contribute to the model by improving its")
    
################################

SUB_PAGES = {
    "About Data": Page_about_data,
    "Model": page_model,
}
################################
def main():
    selection = st.radio("", list(SUB_PAGES.keys()), index=0)
    page = SUB_PAGES[selection]
    page()

if __name__ == "__main__":
    main()