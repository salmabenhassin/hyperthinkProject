import os
import uvicorn
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Imports internes
from src.config import Config
from src.contextual import ContextualProcessor
from src.vector_store import HybridStore
from src.models import ModelFactory

# Configuration
TARGET_PDF = "data/attention_is_all_you_need.pdf"
INDEX_FOLDER = "data/faiss_index"

app = FastAPI(title="RAG : Attention Is All You Need", version="2.0")

# Initialisation globale
rag_store = HybridStore()
llm_client = ModelFactory.get_llm()

# --- Modèles Pydantic ---
class QueryRequest(BaseModel):
    q: str
    k: int = 6

class SourceItem(BaseModel):
    chunk_id: int
    score: float
    method: str
    preview: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]

# --- Logique d'Ingestion Automatique ---
# Dans main.py

def ingest_local_file():
    """
    Lit le fichier PDF local, le découpe, génère les embeddings et sauvegarde l'index.
    """
    print(f"🔄 Démarrage de l'ingestion pour : {TARGET_PDF}")
    
    if not os.path.exists(TARGET_PDF):
        print(f"❌ ERREUR CRITIQUE : Le fichier {TARGET_PDF} est introuvable !")
        return False

    processor = ContextualProcessor()
    
    # 1. Chargement et découpage
    print("✂️ Lecture et découpage du PDF...")
    raw_docs, basic_chunks = processor.load_and_split(TARGET_PDF)
    
    # 2. Contextualisation (Attention au Rate Limit Cohere !)
    print(f"🧠 Génération des embeddings pour {len(basic_chunks)} segments...")
    
    # Pause de sécurité pour éviter l'erreur 429 si nécessaire
    # time.sleep(2) 
    
    contextualized_chunks = processor.generate_contextual_chunks(
        "Ce document présente l'architecture Transformer.", 
        basic_chunks
    )
    
    # 3. Construction et Sauvegarde
    print("💾 Création et sauvegarde de l'index FAISS...")
    
    # --- CORRECTION ICI : On ajoute le deuxième argument (filename) ---
    rag_store.build_index(contextualized_chunks, os.path.basename(TARGET_PDF)) 
    # -----------------------------------------------------------------
    
    return True

# --- Événement de Démarrage (Le Cœur du Système) ---
@app.on_event("startup")
async def startup_event():
    print("🚀 Initialisation du serveur RAG...")
    
    # Étape 1 : Essayer de charger l'index existant
    if rag_store.load_index():
        print("✅ Index FAISS chargé depuis le disque. Le système est prêt instantanément !")
        # Petite vérification optionnelle : est-ce qu'il y a des docs ?
        if rag_store.vector_db:
            print(f"   -> Base vectorielle active.")
    
    # Étape 2 : Si pas d'index, on le crée
    else:
        print("⚠️ Aucun index trouvé. Lancement de l'ingestion automatique...")
        success = ingest_local_file()
        if success:
             print("✅ Ingestion terminée avec succès. Le système est prêt.")
        else:
             print("❌ Échec de l'initialisation. Vérifiez que le fichier PDF est bien dans /data.")

# --- Endpoint de Question (Lecture Seule) ---
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not rag_store.vector_db:
        raise HTTPException(status_code=503, detail="Le système est en cours d'initialisation ou l'index est vide.")

    try:
        # 1. Recherche
        retrieved_docs = rag_store.search(request.q, k=request.k)
        
        # 2. Prompt Système Strict
# Dans main.py, à l'intérieur de query_endpoint
        
        system_prompt = (
            "You are an expert on the research paper 'Attention Is All You Need'. "
            "Answer the user's question using ONLY the context provided below. "
            "Strict rules:\n"
            "1. Answer in English.\n"
            "2. If the answer is not in the context, say exactly: 'I cannot answer this based on the provided context.'\n"
            "3. Do not use outside knowledge.\n"
            "4. Always cite the source index (e.g., [Source 1])."
        )
        context_str = "\n\n".join([f"[Source {i+1}] {d['content']}" for i, d in enumerate(retrieved_docs)])

        # 3. Génération
        response = llm_client.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXTE:\n{context_str}\n\nQUESTION:\n{request.q}"}
        ])

        # 4. Formatage
        sources_output = []
        for doc in retrieved_docs:
            c_id = doc.get("chunk_id", 0) # Sécurité int
            sources_output.append({
                "chunk_id": int(c_id) if c_id is not None else 0,
                "score": float(doc.get("score", 0.0)),
                "method": str(doc.get("source_method", "Unknown")),
                "preview": str(doc.get("content", ""))[:80] + "..."
            })

        return {"answer": response.content, "sources": sources_output}

    except Exception as e:
        print(f"Erreur Query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)