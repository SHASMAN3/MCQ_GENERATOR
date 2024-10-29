# src/mcqgenerator/MCQGenerator.py

import os
from dotenv import load_dotenv
import json
from typing import Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Dict, Optional

# Load environment variables
load_dotenv()

# Load response template
with open('Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# Define the response schema to match your JSON format
class QuizQuestion(BaseModel):
    mcq: str = Field(description="The question text")
    options: Dict[str, str] = Field(description="Dictionary of options with keys a, b, c, d")
    correct: str = Field(description="The correct answer")

class QuizResponse(BaseModel):
    questions: Dict[str, QuizQuestion]

# Initialize ChatOpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in your environment variables.")

chat_model = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model_name="gpt-3.5-turbo",
    temperature=0.5
)

# Templates
TEMPLATE1 = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming to the text as well.
Make sure to format your response exactly like the JSON format below and use it as a guide. \
Ensure to make exactly {number} MCQs.

The response should be a valid JSON object with numbered questions (1, 2, 3, etc.).
Each question should have the following format:
{{
    "mcq": "Question text here",
    "options": {{
        "a": "First option",
        "b": "Second option",
        "c": "Third option",
        "d": "Fourth option"
    }},
    "correct": "correct answer letter (a, b, c, or d)"
}}

### Example Response Format:
{response_json}

Remember to make your response parseable as JSON.
"""

TEMPLATE2 = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. \
Only use at max 50 words for complexity analysis. 
If the quiz is not at par with the cognitive and analytical abilities of the students,\
update the quiz questions which need to be changed and change the tone such that it perfectly fits the student abilities.

Quiz_MCQs:
{quiz}

Provide your review in a way that can be easily parsed as a string.
"""

def create_mcq_chain(subject: str, number: int, tone: str, file_path: str) -> Dict[str, Any]:
    """
    Creates and executes the MCQ generation and evaluation chain.
    """
    try:
        # Read input text
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Create prompts
        quiz_generation_prompt = ChatPromptTemplate.from_template(template=TEMPLATE1)
        quiz_evaluation_prompt = ChatPromptTemplate.from_template(template=TEMPLATE2)
        
        # Create the quiz generation chain
        quiz_chain = (
            {
                "text": RunnablePassthrough(),
                "number": lambda _: number,
                "subject": lambda _: subject,
                "tone": lambda _: tone,
                "response_json": lambda _: json.dumps(RESPONSE_JSON, indent=2)
            }
            | quiz_generation_prompt
            | chat_model
            | StrOutputParser()
        )

        # Execute quiz generation and ensure JSON format
        quiz_result = quiz_chain.invoke(text)
        
        # Try to parse the result as JSON
        try:
            quiz_json = json.loads(quiz_result)
            # Validate the structure matches our expected format
            if not all(key in quiz_json[str(i)] for i in range(1, len(quiz_json) + 1) 
                      for key in ['mcq', 'options', 'correct']):
                raise ValueError("Quiz response is missing required fields")
        except json.JSONDecodeError:
            # If parsing fails, try to extract JSON from the response
            import re
            json_match = re.search(r'\{[\s\S]*\}', quiz_result)
            if json_match:
                quiz_json = json.loads(json_match.group())
            else:
                raise ValueError("Could not extract valid JSON from the response")

        # Create and execute the review chain
        review_chain = (
            quiz_evaluation_prompt
            | chat_model
            | StrOutputParser()
        )

        review_result = review_chain.invoke({
            "subject": subject,
            "quiz": json.dumps(quiz_json, indent=2)
        })

        # Return the results in a consistent format
        return {
            "quiz": quiz_json,
            "review": review_result
        }

    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error in MCQ generation: {str(e)}")