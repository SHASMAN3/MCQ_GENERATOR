from setuptools import find_packages,setup


setup(
    name='mcqgenerator',
    version='0.0.1',
    author="shashim mankar",
    author_email="shashimmnkar6@gmail.com",
    install_requires=["openai","langchain","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages()
    
    
    
    
)# if we put -e . at the end of all libraties in requirements.txt it find automatically local packages and install it.