# pip dependencies: python-dotenv
import os
from dotenv import load_dotenv, dotenv_values
from aicluster import GeminiAI

from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize dataset
dataset_name = "databricks/databricks-dolly-15k"
page_content_column = "response"

# Initialize dataset Loader using HuggingFaceDatasetLoader
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)

data = loader.load()

# data[:2]

# Text Chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

docs = text_splitter.split_documents(data)

