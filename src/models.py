from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import Config

class ModelFactory:
    @staticmethod
    def get_llm():
        """Retourne une instance de Cohere Chat Model"""
        if not Config.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY manquante dans .env")
        
        return ChatCohere(
            model=Config.LLM_MODEL_NAME,
            cohere_api_key=Config.COHERE_API_KEY,
            temperature=0
        )

    @staticmethod
    def get_embeddings():
        # On peut garder les embeddings gratuits ou passer à CohereEmbeddings
        return HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)