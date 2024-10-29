import PyPDF2
import json
from typing import Dict, List, Any

def read_file(file) -> str:
    """
    Read text from uploaded file (PDF or TXT)
    
    Args:
        file: Uploaded file object from Streamlit
        
    Returns:
        str: Extracted text from the file
    """
    text = ""
    
    try:
        # Get the file name
        file_name = file.name
        
        # Check file type and process accordingly
        if file_name.endswith('.pdf'):
            # Read PDF file
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file_name.endswith('.txt'):
            # Read text file
            text = file.getvalue().decode('utf-8')
        else:
            raise ValueError("Unsupported file format. Please upload a PDF or TXT file.")
            
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def get_table_data(quiz_str: str) -> List[Dict[str, Any]]:
    """
    Convert quiz data into a format suitable for DataFrame
    
    Args:
        quiz_str: Quiz data either as string or dictionary
        
    Returns:
        List[Dict]: List of dictionaries containing formatted quiz data
    """
    try:
        # Parse the quiz string if it's a string
        if isinstance(quiz_str, str):
            quiz_data = json.loads(quiz_str)
        else:
            quiz_data = quiz_str
            
        # Extract questions data
        if isinstance(quiz_data, dict) and "questions" in quiz_data:
            questions = quiz_data["questions"]
        else:
            questions = quiz_data
            
        table_data = []
        
        for i, question in enumerate(questions, 1):
            options = question.get("options", {})
            table_data.append({
                "Question": question.get("question", ""),
                "Options": "\n".join([f"{k}) {v}" for k, v in options.items()]),
                "Correct Answer": question.get("correct_answer", ""),
                "Explanation": question.get("explanation", "")
            })
            
        return table_data
        
    except Exception as e:
        raise Exception(f"Error processing quiz data: {str(e)}")