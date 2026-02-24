import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    PDF_PATH = "data/attention_is_all_you_need.pdf"
    
    # --- CHANGEMENT ICI ---
    # Clé API (à mettre dans votre .env : COHERE_API_KEY=...)
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    
    # Modèle : 'command-r-plus' est le plus puissant, 'command-r' est plus rapide/moins cher
    LLM_MODEL_NAME = "command-r-plus-08-2024" 
    
    # Optionnel : Utiliser les embeddings Cohere (meilleurs pour l'anglais/multilingue)
    # ou garder HuggingFace. Ici on garde HuggingFace pour économiser les crédits API.
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    FAISS_INDEX_PATH = "data/faiss_index" # Dossier où l'index sera stocké
    INDEX_METADATA_FILE = "data/faiss_index/metadata.json"
    VECTOR_DB_PATH = "data/chroma_db"
    COLLECTION_NAME = "research_paper_cohere"