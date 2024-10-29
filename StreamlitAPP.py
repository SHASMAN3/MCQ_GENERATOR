# app.py

import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, List
import streamlit as st
from langchain.callbacks import get_openai_callback

from src.mcqgenerator.MCQGenerator import create_mcq_chain
from src.mcqgenerator.utils import read_file
from src.mcqgenerator.logger import logging

# Load environment variables
load_dotenv()

def get_table_data(quiz_dict: Dict) -> List[Dict]:
    """
    Convert the quiz dictionary to a format suitable for DataFrame
    """
    table_data = []
    
    for question_num, question_data in quiz_dict.items():
        row = {
            'Question': question_data['mcq'],
            'Options': "\n".join([f"{k}: {v}" for k, v in question_data['options'].items()]),
            'Correct Answer': question_data['correct'].upper()
        }
        table_data.append(row)
    
    return table_data

def create_mcq_app():
    """
    Creates the Streamlit application for MCQ generation
    """
    st.title("⛓ 🦜 MCQ Creator Application with LangChain 🦜 ⛓")
    
    with st.form("user_inputs"):
        uploaded_file = st.file_uploader("Upload a PDF or txt file")
        mcq_count = st.number_input("Number of MCQs", min_value=3, max_value=50, value=5)
        subject = st.text_input("Insert subject", max_chars=20)
        tone = st.text_input("Complexity level of questions", max_chars=20, 
                            placeholder="simple/medium/complex")
        
        button = st.form_submit_button("Create MCQs")
        
        if button and uploaded_file is not None and mcq_count and subject and tone:
            with st.spinner("Generating MCQs..."):
                try:
                    # Read the uploaded file
                    text = read_file(uploaded_file)
                    
                    # Create a temporary file path for the text content
                    temp_file_path = "temp_input.txt"
                    with open(temp_file_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    # Count tokens and track the cost of the API call
                    with get_openai_callback() as cb:
                        # Generate MCQs using the create_mcq_chain function
                        response = create_mcq_chain(
                            subject=subject,
                            number=mcq_count,
                            tone=tone,
                            file_path=temp_file_path
                        )
                        
                        # Clean up temporary file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                        
                        # Display token usage and cost
                        st.sidebar.subheader("API Usage Statistics")
                        st.sidebar.write(f"Total Tokens: {cb.total_tokens}")
                        st.sidebar.write(f"Prompt Tokens: {cb.prompt_tokens}")
                        st.sidebar.write(f"Completion Tokens: {cb.completion_tokens}")
                        st.sidebar.write(f"Total Cost: ${cb.total_cost:.4f}")
                        
                        # Process the response
                        if isinstance(response, dict) and "quiz" in response:
                            quiz_data = response["quiz"]
                            if quiz_data:
                                # Convert quiz data to DataFrame
                                table_data = get_table_data(quiz_data)
                                if table_data:
                                    st.subheader("Generated MCQs")
                                    df = pd.DataFrame(table_data)
                                    df.index = df.index + 1
                                    st.table(df)
                                    
                                    # Display the review
                                    st.subheader("Quiz Review")
                                    st.text_area(
                                        label="Expert Review", 
                                        value=response.get("review", "No review available"),
                                        height=150,
                                        disabled=True
                                    )
                                else:
                                    st.error("Failed to process quiz data into table format")
                            else:
                                st.error("No questions generated in the quiz")
                        else:
                            st.error("Invalid response format from MCQ generator")
                            st.error(f"Response received: {response}")
                            
                except Exception as e:
                    st.error("An error occurred during MCQ generation")
                    st.error(str(e))
                    logging.error(f"Error in MCQ generation: {str(e)}")
                    logging.error(traceback.format_exc())
                finally:
                    # Ensure temporary file is cleaned up even if an error occurs
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

if __name__ == "__main__":
    create_mcq_app()