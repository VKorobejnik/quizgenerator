import os

# Supported languages
SUPPORTED_LANGUAGES = {
    "English": "en",
    "German": "de",
    "Polski": "pl",
    "Romanian": "ro",
    "Български": "bg"
}

API_KEY = os.getenv("DEEPSEEK_API_KEY") #os.getenv("OPENAI_API_KEY") 
BASE_URL = "https://api.deepseek.com"


# Model configurations
MULTILINGUAL_EMBEDDING_MODEL = "distiluse-base-multilingual-cased"
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
LLM_MODEL= "deepseek-chat"  #"gpt-4o" 

# Chunking parameters
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200
DESIRED_TOPICS = 10
MIN_CHUNKS = DESIRED_TOPICS + 5
MAX_CHUNK_SIZE = 8000
MIN_CHUNK_SIZE = 500
