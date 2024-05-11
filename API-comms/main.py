# pip dependencies: python-dotenv
import os
from dotenv import load_dotenv, dotenv_values
from aicluster import GeminiAI

from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

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

# Define embedding model
modelPath = "sentence-transformers/all-MiniLM-l6-v2"
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

# Load into database (FAISS)
db = FAISS.from_documents(docs, embeddings)

question = "Why can camels survive for long without water?"
searchDB = db.similarity_search(question)
print(searchDB[0].page_content)



# def prep():
#     load_dotenv()

# def main():
#     prep()
#     aiobj = GeminiAI(os.getenv("GEMINI_API_KEY"))
#     # aiobj.get_response_default("Who was elected the president of the US in 2012")

# if __name__ == "__main__":
#     main()
