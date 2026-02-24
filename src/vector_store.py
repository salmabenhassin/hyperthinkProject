import os
import json  # <--- C'était l'import manquant !
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document

from src.models import ModelFactory
from src.config import Config

class HybridStore:
    def __init__(self):
        self.embeddings = ModelFactory.get_embeddings()
        self.vector_db = None
        self.bm25_retriever = None
        # Chemins fixes pour la persistance
        self.index_path = "data/faiss_index"
        self.metadata_path = "data/faiss_index/metadata.json"
        
        try:
            self.reranker = CohereRerank(
                cohere_api_key=Config.COHERE_API_KEY, 
                model="rerank-english-v3.0"
            )
        except Exception as e:
            print(f"⚠️ Cohere Rerank non dispo: {e}")
            self.reranker = None

    def get_indexed_filename(self):
        """Récupère le nom du fichier actuellement stocké sur le disque."""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                return json.load(f).get("filename")
        return None

    def build_index(self, documents: List[Document], filename: str):
        """Construit, sauvegarde l'index et enregistre le nom du fichier."""
        print(f"🏗️ Ingestion complète pour {filename}...")
        
        # 1. Création FAISS
        self.vector_db = FAISS.from_documents(documents, self.embeddings)
        
        # 2. Sauvegarde Locale
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
            
        self.vector_db.save_local(self.index_path)
        
        # 3. Sauvegarde des métadonnées (C'est ici que ça plantait)
        with open(self.metadata_path, "w") as f:
            json.dump({"filename": filename}, f)
            
        # 4. Création BM25
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        print(f"✅ Index sauvegardé pour {filename}")

    def load_index(self, documents: List[Document] = None):
        """Charge l'index existant et initialise BM25."""
        if os.path.exists(self.index_path):
            try:
                self.vector_db = FAISS.load_local(
                    self.index_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                # BM25 doit être recréé à partir des documents car il ne se sauvegarde pas seul
                # Note : Si documents est None, BM25 ne sera pas actif (recherche vectorielle pure)
                if documents:
                    self.bm25_retriever = BM25Retriever.from_documents(documents)
                
                return True
            except Exception as e:
                print(f"❌ Erreur chargement: {e}")
        return False

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Recherche hybride avec Reranking."""
        if not self.vector_db:
            raise ValueError("L'index n'est pas prêt.")

        fetch_k = k * 2
        
        # A. Recherche Vectorielle (FAISS)
        vector_docs = self.vector_db.similarity_search_with_score(query, k=fetch_k)
        
        # B. Recherche BM25 (Si disponible)
        bm25_docs = []
        if self.bm25_retriever:
            self.bm25_retriever.k = fetch_k
            bm25_docs = self.bm25_retriever.invoke(query)

        # C. Fusion
        unique_docs = {}
        for doc, score in vector_docs:
            doc.metadata["retrieval_source"] = "vector (faiss)"
            doc.metadata["original_score"] = float(score)
            unique_docs[doc.metadata.get("chunk_id")] = doc
            
        for doc in bm25_docs:
            c_id = doc.metadata.get("chunk_id")
            if c_id not in unique_docs:
                doc.metadata["retrieval_source"] = "bm25"
                doc.metadata["original_score"] = 0.0
                unique_docs[c_id] = doc
            else:
                unique_docs[c_id].metadata["retrieval_source"] = "hybrid"

        candidates = list(unique_docs.values())
        
        # D. Reranking (Cohere)
        if self.reranker:
            try:
                reranked = self.reranker.compress_documents(candidates, query=query)
                formatted = []
                for res in reranked[:k]:
                    # Récupération blindée du score
                    score_val = getattr(res, "relevance_score", 0.0)
                    if score_val == 0.0:
                        score_val = res.metadata.get("relevance_score", 0.0)

                    formatted.append({
                        "chunk_id": res.metadata.get("chunk_id"),
                        "content": res.page_content,
                        "score": round(float(score_val), 4),
                        "source_method": res.metadata.get("retrieval_source", "hybrid_reranked")
                    })
                return formatted
            except Exception as e:
                print(f"⚠️ Erreur Rerank: {e}")

        # Fallback si pas de Rerank
        return [{
            "chunk_id": d.metadata.get("chunk_id"), 
            "content": d.page_content, 
            "score": 0.0, 
            "source_method": d.metadata.get("retrieval_source")
        } for d in candidates[:k]]