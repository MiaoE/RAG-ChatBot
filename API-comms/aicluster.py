"""AI Cluster: This file hosts the main objects for compatible generative AI tools
"""
# pip dependencies: google-generativeai, 
import google.generativeai as genai

class GeminiAI:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # text only, gemini-pro-vision: text and image
        self.model = genai.GenerativeModel('gemini-pro')

    def get_response_default(self, query:str) -> str:
        response = self.model.generate_content(query)
        print(response.text)
        return response.text