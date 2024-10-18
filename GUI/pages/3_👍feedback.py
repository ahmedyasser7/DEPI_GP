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


################################
# Main app structure
st.set_page_config(page_title="Feedback", layout="wide", page_icon= r"GUI\home_images\good-feedback.png")

################################
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
def main():
    page = Page_feedback
    page()

if __name__ == "__main__":
    main()