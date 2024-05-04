# pip dependencies: argparse, google-generativeai, python-dotenv
import os#, argparse
import google.generativeai as genai
from dotenv import load_dotenv, dotenv_values

class GeminiAI:
    def __init__(self):
        genai.configure(api_key=os.getenv("API_KEY"))
        # text only, gemini-pro-vision: text and image
        self.model = genai.GenerativeModel('gemini-pro')

    def get_response_default(self, query:str) -> str:
        response = self.model.generate_content(query)
        print(response.text)
        return response.text


def prep():
    load_dotenv()

def main():
    prep()
    aiobj = GeminiAI()
    # aiobj.get_response_default("Who was elected the president of the US in 2012")

if __name__ == "__main__":
    main()

