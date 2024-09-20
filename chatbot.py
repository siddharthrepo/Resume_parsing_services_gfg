from pathlib import Path as p
from pprint import pprint
import pandas as pd
from PyPDF2 import PdfReader
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv('/home/siddharth/Desktop/resume_parser/.env')

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-pro" , google_api_key = GOOGLE_API_KEY , temperature = 0.3 , convert_system_message_to_human = True)

