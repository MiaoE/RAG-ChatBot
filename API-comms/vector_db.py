# pip dependency: pinecone-client
from pinecone import Pinecone, ServerlessSpec

class VectorDB:
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)
    
    def create_index(self):
        self.pc.create_index(
            name="quickstart",
            dimension=8, # Replace with your model dimensions
            metric="euclidean", # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
