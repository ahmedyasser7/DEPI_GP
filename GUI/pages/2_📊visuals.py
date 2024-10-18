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
st.set_page_config(page_title="Some Of Data Art", layout="wide", page_icon=r"GUI/home_images/web-analytics.png")
################################

def page_visuals():
    st.title("Visuals")
    BASE_DIR = os.path.dirname(__file__)  # Current directory where the script exists
    ANALYSIS_IMAGE_DIR = os.path.join(BASE_DIR, "analysis_images")  # Set the path to the analysis images directory

    images = [
        os.path.join(ANALYSIS_IMAGE_DIR, "correlation.png"),
        os.path.join(ANALYSIS_IMAGE_DIR, "HeatmapofAccidentSeveritybyRoadSurface.png"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0001.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0002.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0003.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0004.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0005.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0006.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0007.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0008.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0009.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0010.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0011.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0012.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "IMG-20241018-WA0013.jpg"),
        os.path.join(ANALYSIS_IMAGE_DIR, "pie.png"),
        os.path.join(ANALYSIS_IMAGE_DIR, "rural_vs_urban.png"),
        os.path.join(ANALYSIS_IMAGE_DIR, "urban_vs_rural.png"),
    ]

    # CSS styling for the buttons and slider indicators
    st.markdown("""
        <style>
        .btn-style {
            background-color: #851CA3;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease-in-out;
        }
        .btn-style:hover {
            background-color: #AA0055;
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

    # Load the image function
    def load_image(image_path):
        try:
            print(f"Loading image from: {image_path}")  # Debugging line
            return Image.open(image_path)
        except FileNotFoundError:
            st.error(f"Image not found: {image_path}")  # Improved error message
            return None

    # Initialize session state for image index
    if "carousel_index" not in st.session_state:
        st.session_state.carousel_index = 0

    # Display the current image
    image = load_image(images[st.session_state.carousel_index])
    if image is not None:  # Only display if the image was loaded successfully
        st.image(image, width=700)

    # Navigation buttons (Previous/Next)
    prev, _, next = st.columns([1, 10, 1])

    # Handle the previous button click
    if prev.button("⬅️", key="prev", help="Previous image", type="primary"):
        st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(images)

    # Handle the next button click
    if next.button("➡️", key="next", help="Next image", type="primary"):
        st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(images)

    # Display current image index
    st.write(f"Image {st.session_state.carousel_index + 1} of {len(images)}")

    st.divider()
    
################################
def main():
    page = page_visuals
    page()

if __name__ == "__main__":
    main()
    
    
###############################################################
# Emergency assistance sub-pages:

# # Questions for the visuals
# questions = [
#     "How does the weather impact the number or severity of an accident?",
#     "Does driver age have an effect on the number of accidents?",
#     "What is the relation between hour, day, week, and month with several fatal accidents?",
#     "Are certain car models safer than others?",
#     "Is the social class of a casualty dependent on the accident severity?",
#     "Can you forecast the future daily/weekly/monthly accidents?",
#     "What about fatal accidents—can you predict them?",
#     "Can you predict if an accident was fatal? (like Titanic prediction)"
# # ]

# selected_question = st.selectbox("Please, choose a question", questions)

#     question_map = {
#         questions[0]: 2,
#         questions[1]: 3,
#         questions[2]: 4,
#         questions[3]: 5,
#         questions[4]: 6,
#         questions[5]: 7,
#         questions[6]: 8,
#         questions[7]: 9,
#     }

#     # cell_index = question_map[selected_question]
#     # No notebook visuals are displayed here, just the question mapping.

