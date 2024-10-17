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
import pickle
import os
###############################
HOME_BASE_DIR = os.path.dirname(__file__)  # Current directory where Bandass.py exists
HOME_IMAGE_DIR = os.path.join(HOME_BASE_DIR, "home_images")  # Set the path to the images directory

################################
# Main app structure
st.set_page_config(page_title="Bandaas", layout="wide", page_icon= r"GUI/home_images/accident-car.png")

@st.cache_resource
def load_image(image_path):
    try:
        print(f"Loading image from: {image_path}")  # Debugging line
        return Image.open(image_path)
    except FileNotFoundError:
        st.error(f"Image not found: {image_path}")  # Improved error message
        return None
################################
@st.cache_resource
def page_image_display():
    st.title("LOOK OUT!!!")
    image = load_image(os.path.join(HOME_IMAGE_DIR, "two_cars.png"))  # Load the image
    if image is not None:  # Check if the image loaded successfully
        st.image(image, caption="We are a collaborative Team!")
        
################################
@st.cache_resource
def Page_overview():
    st.title("Overview")
    st.write(f"The link for the GitHub Repo")
    st.write(f"The link for the Documentation")
    st.write(f"The link for the Presentation")


################################
@st.cache_resource
def page_authors():
    st.title("Teammates")
    st.write("## Names")
    st.write("""
    - Ahmed Yasser 
    - Ahmed Abd El-Hameed (7ameedo)
    - Abram Maher
    - Sarah Mohammed Selim
    - Naglaa Reda
    """)
    st.write(" We wish you enjoy this journey!")

################################
Bandass_PAGES = {
    "Hello": page_image_display,
    "Overview": Page_overview,
    "Team Names": page_authors,
}

################################
def main():
    selection = st.radio("", list(Bandass_PAGES.keys()), index=0)
    page = Bandass_PAGES[selection]
    page()

if __name__ == "__main__":
    main()
