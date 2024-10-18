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
st.set_page_config(page_title="Some Of Data Art", layout="wide", page_icon=r"GUI/home_images/web-analytics.png")
################################

def page_visuals():
    st.title("Visuals")
    images = [
        r"GUI\pages\analysis_images\correlation.png",
        r"GUI\pages\analysis_images\HeatmapofAccidentSeveritybyRoadSurface.png",
        r"GUI\pages\analysis_images\IMG-20241018-WA0001.jpg",
        r"GUI\pages\analysis_images\IMG-20241018-WA0002.jpg",
        r"GUI\pages\analysis_images\IMG-20241018-WA0003.jpg",
        r"GUI\pages\analysis_images\IMG-20241018-WA0004.jpg",
        r"GUI\pages\analysis_images\IMG-20241018-WA0005.jpg",
        r"GUI\pages\analysis_images\IMG-20241018-WA0006.jpg",
        r"GUI\pages\analysis_images\IMG-20241018-WA0007.jpg",
        r"GUI\pages\analysis_images\IMG-20241018-WA0008.jpg",
        r"GUI\pages\analysis_images\IMG-20241018-WA0009.jpg",
        r"GUI\pages\analysis_images\IMG-20241018-WA0010.jpg",
        r"GUI\pages\analysis_images\IMG-20241018-WA0011.jpg",
        r"GUI\pages\analysis_images\IMG-20241018-WA0012.jpg",
        r"GUI\pages\analysis_images\IMG-20241018-WA0013.jpg",
        r"GUI\pages\analysis_images\pie.png",
        r"GUI\pages\analysis_images\rural_vs_urban.png",
        r"GUI\pages\analysis_images\urban_vs_rural.png",
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

    if "carousel_index" not in st.session_state:
        st.session_state.carousel_index = 0

    st.image(images[st.session_state.carousel_index], width=700)

    prev, _, next = st.columns([1, 10, 1])
    
    if prev.button("⬅️", key="prev", help="Previous image", type="primary"):
        st.session_state.carousel_index = (st.session_state.carousel_index - 1) % len(images)

    if next.button("➡️", key="next", help="Next image", type="primary"):
        st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(images)

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

