import streamlit as st
import pandas as pd
from openpyxl import load_workbook
import os

st.set_page_config(page_title="Feedback", layout="wide", page_icon="GUI/home_images/good-feedback.png")

def save_feedback_to_excel(feedback):
    file_path = "feedbacks.xlsx"
    feedback_df = pd.DataFrame({"Feedback": [feedback]})

    try:
        if os.path.exists(file_path):
            existing_data = pd.read_excel(file_path)
            combined_data = pd.concat([existing_data, feedback_df], ignore_index=True)
        else:
            combined_data = feedback_df

        with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
            combined_data.to_excel(writer, index=False)

    except Exception as e:
        st.error(f"An error occurred while saving feedback: {str(e)}")

def Page_feedback():
    st.title("Your Feedback")
    user_input = st.text_input("Enter your feedback or data")
    if st.button("Submit"):
        if user_input:
            save_feedback_to_excel(user_input)
            st.write("Thank you for your feedback!")
        else:
            st.warning("Please enter your feedback before submitting!")

def main():
    Page_feedback()

if __name__ == "__main__":
    main()