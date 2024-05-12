# pip dependencies: python-dotenv
import os
from dotenv import load_dotenv, dotenv_values
from API_testing.aicluster import GeminiAI

from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModel, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA

## TODO: re-format this file so it runs continuously (for a front-end UI)

# Initialize dataset
dataset_name = "databricks/databricks-dolly-15k"
page_content_column = "response"

## TODO: feed more data for more accurate and current results. Have a list of datasets and parse them into a feeder
## TODO: implement real-time data for most current results
## TODO: have multiple data types for embedding, perhaps could use Langchain's PDF Loader library

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

## TODO: Deploy vector database to cloud instead of local

# Initialize LLM
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

model_name = "deepset/roberta-base-squad2"

question_answer_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)

## TODO: fine-tuning the model and LLM

llm = HuggingFacePipeline(
    pipeline=question_answer_pipeline, 
    model_kwargs={"temperature": 0.5, "max_length": 512},
)

query = "Where do camels get their energy from?"

# qa_chain = load_qa_chain(llm, chain_type="stuff")
# searchDB_result = db.similarity_search(query)
# result = qa_chain.run(input_documents=searchDB_result, question=query)
# print(result["result"])

retriever = db.as_retriever()
# combine_docs_chain = create_stuff_documents_chain(llm, query)
# retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
result = result = qa.run({"query": query})
print(result["result"])

# def prep():
#     load_dotenv()

# def main():
#     prep()
#     aiobj = GeminiAI(os.getenv("GEMINI_API_KEY"))
#     # aiobj.get_response_default("Who was elected the president of the US in 2012")

# if __name__ == "__main__":
#     main()

## TODO: a front-end UI to interact with the ChatBot