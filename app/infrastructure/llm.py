import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_llm_client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_available_gpt_models():
    try:
        client = get_llm_client()
        models_response = client.models.list()
        gpt_models = sorted([
            model.id for model in models_response.data 
            if model.id.startswith('gpt')
        ])
        if not gpt_models:
            gpt_models = ["gpt-5-mini", "gpt-4", "gpt-3.5-turbo"]  # fallback
        return gpt_models
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["gpt-5-mini", "gpt-4", "gpt-3.5-turbo"]  # fallback
