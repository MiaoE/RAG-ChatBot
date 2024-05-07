# pip dependencies: python-dotenv
import os
from dotenv import load_dotenv, dotenv_values
from aicluster import GeminiAI

def prep():
    load_dotenv()

def main():
    prep()
    aiobj = GeminiAI(os.getenv("GEMINI_API_KEY"))
    # aiobj.get_response_default("Who was elected the president of the US in 2012")

if __name__ == "__main__":
    main()
