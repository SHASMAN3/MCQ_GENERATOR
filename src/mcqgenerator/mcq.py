import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

# Import necessary packages from LangChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from the .env file
load_dotenv()

# Access the environment variables just like you would with os.environ
key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=key, model_name="gpt-3.5-turbo", temperature=0.7)


# prompt = ChatPromptTemplate.from_template("tell me a sentence about {politician}")

# chain = prompt | model | StrOutputParser()

# Quiz generation prompt
quiz_template = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. Make sure the questions are not repeated and check all the questions to conform to the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide.
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""

quiz_generation_prompt = ChatPromptTemplate.from_template(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=quiz_template
)

# Quiz evaluation prompt
evaluation_template = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students.
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. If the quiz is not up to the cognitive and analytical abilities of the students,
update the quiz questions which need to be changed and change the tone such that it perfectly fits the student's abilities.
Quiz_MCQs: {quiz}
Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = ChatPromptTemplate.from_template(
    input_variables=["subject", "quiz"],
    template=evaluation_template
)

# Create the chains
prompt | model | StrOutputParser()
quiz_chain = quiz_generation_prompt | llm | StrOutputParser()
review_chain = quiz_evaluation_prompt | llm | StrOutputParser() 

# Define inputs and outputs for SequentialChain
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True
)

