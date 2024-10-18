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
    st.write(f"The link for the GitHub Repo: https://github.com/ahmedyasser7/DEPI_GP")
    st.write(f"The link for the Documentation and the presentaion: {4}")


################################
@st.cache_resource
def page_authors():
    st.markdown("""
            <style>
            .author-container {
                background-color: #111111;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .author-title {
                color: #FF6F61;
                text-align: center;
                margin-bottom: 20px;
            }
            .author-list {
                list-style-type: none;
                padding: 0;
            }
            .author-list li {
                font-size: 1.2em;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
            }
            .author-list li a {
                margin-left: 10px;
                color: #0072b1; /* LinkedIn blue */
                text-decoration: none;
            }
            .author-list li a:hover {
                color: #005582; /* Darker LinkedIn blue on hover */
            }
            .linkedin-icon {
                width: 20px;
                height: 20px;
            }
            .footer {
                text-align: center;
                font-size: 1.2em;
                margin-top: 30px;
            }
            </style>
            """, unsafe_allow_html=True)

    st.markdown("<h1 class='author-title'>Meet Our Teammates</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="author-container">
        <ul class="author-list">
            <li>
                <strong>Ahmed Yasser Taha</strong>
                <a href="https://www.linkedin.com/in/ahmedyassertaha/" target="_blank">
                    <img class="linkedin-icon" src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
                    LinkedIn
                </a>
            </li>
            <li>
                <strong>Ahmed AbdulHameed Mahmoud</strong>
                <a href="https://www.linkedin.com/in/ahmed-abdulhameed-067871239/" target="_blank">
                    <img class="linkedin-icon" src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
                    LinkedIn
                </a>
            </li>
            <li>
                <strong>Abram Maher Samwel</strong>
                <a href="https://www.linkedin.com/in/engabrammaher/" target="_blank">
                    <img class="linkedin-icon" src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
                    LinkedIn
                </a>
            </li>
            <li>
                <strong>Sarah Mohammed Selim</strong>
                <a href="https://www.linkedin.com/in/sarah-mohamed-selim-a57aa0284/" target="_blank">
                    <img class="linkedin-icon" src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
                    LinkedIn
                </a>
            </li>
            <li>
                <strong>Naglaa Reda Ali</strong>
                <a href="https://www.linkedin.com/in/naglaa-reda-6874452a2/" target="_blank">
                    <img class="linkedin-icon" src="https://cdn-icons-png.flaticon.com/512/174/174857.png" alt="LinkedIn">
                    LinkedIn
                </a>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='footer'>We hope you enjoy this journey!</div>", unsafe_allow_html=True)


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
