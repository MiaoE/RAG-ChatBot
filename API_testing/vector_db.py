# pip dependency: pinecone-client, transformers, langchain, langchain-openai
from pinecone import Pinecone, ServerlessSpec
from transformers import GPT2TokenizerFast, AutoModel, pipeline
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

## Embedding model code for HuggingFace is too complex
class EmbeddingModel_HuggingFace:
    def __init__(self):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("Xenova/text-embedding-ada-002")
        self.model = AutoModel.from_pretrained("gpt2")
        #self.model.resize_token_embeddings(15)

    def get_vector(self, val:str) -> list[int]:
        #token = self.tokenizer.encode(val)  # this is a token
        # use embedding to turn tokens into vectors
        pipe = pipeline('feature-extraction', model=self.model, tokenizer=self.tokenizer)
        return pipe(val)  # each token converts to vector of 768 floats
    
    def decode_token(self, vec:list[int]):
        return self.tokenizer.decode(vec)

## Test with OpenAI, API key does not work
class EmbeddingModel:
    def __init__(self, api):
        self.obj = OpenAIEmbeddings(openai_api_key=api)

    def embed_single(self, val:str):
        return self.obj.embed_query(val)
    
    def embed_multiple(self, vals:list[str]):
        return self.obj.embed_documents(vals)

## Langchain does not support Pinecone
class VectorDB:
    def __init__(self, api_key):
        self.pc = Pinecone(api_key=api_key)
    
    def create_index(self):
        self.pc.create_index(
            name="quickstart",
            dimension=768, # Replace with your model dimensions
            metric="cosine", # Pinecone supports: [euclidean, cosine, dotproduct]
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    def insert_entries(self, vec_name:list[str], vec_values:list[list[float]], ns:str) -> bool:
        '''dimension of vec_name must equal dimension of vec_values: len(vec_name) == len(vec_values)
        Returns True if successfully executed, False otherwise'''
        try:
            assert len(vec_name) == len(vec_values)
        except AssertionError as ex:
            print(ex)
            return False
        new_vectors = []
        for i in range(len(vec_name)):
            new_vectors.append({"id":vec_name[i], "values":vec_values[i]})
        self.pc.upsert(vectors=new_vectors, namespace=ns)  # might be in try catch block

    def query_value(self, ns:str, vector:list[float], top_k:int, include_values:bool):
        return self.pc.query(
            namespace=ns,
            vector=vector,
            top_k=top_k,
            include_values=include_values
        )
        # Returns:
        # {'matches': [{'id': 'vec3',
        #               'score': 0.0,
        #               'values': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]},
        #              {'id': 'vec4',
        #               'score': 0.0799999237,
        #               'values': [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]},
        #              {'id': 'vec2',
        #               'score': 0.0800000429,
        #               'values': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]}],
        #  'namespace': 'ns1',
        #  'usage': {'read_units': 6}}
        # {'matches': [{'id': 'vec7',
        #               'score': 0.0,
        #               'values': [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]},
        #              {'id': 'vec8',
        #               'score': 0.0799999237,
        #               'values': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]},
        #              {'id': 'vec6',
        #               'score': 0.0799999237,
        #               'values': [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]}],
        #  'namespace': 'ns2',
        #  'usage': {'read_units': 6}}


if __name__ == "__main__":
    load_dotenv()
    stuff = EmbeddingModel(os.getenv("OPENAI_API_KEY"))
    print(stuff.embed_single("Willmer is gay"))
    print(stuff.embed_multiple(["What is the weather today", "Oh hello", "How is your day"]))