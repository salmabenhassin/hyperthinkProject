import time  # <--- N'oubliez pas l'import
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.models import ModelFactory

class ContextualProcessor:
    def __init__(self):
        self.llm = ModelFactory.get_llm()

    # def load_and_split(self, file_path: str) -> tuple[List[Document], List[Document]]:
    #     loader = PyPDFLoader(file_path)
    #     raw_docs = loader.load()
    #     splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=1000,
    #         chunk_overlap=200
    #     )
    #     chunks = splitter.split_documents(raw_docs)
    #     return raw_docs, chunks
    def load_and_split(self, file_path: str, strategy: str = "recursive") -> tuple:
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            
            if strategy == "recursive":
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            elif strategy == "token":
                # Stratégie 2 : Découpage par tokens (plus précis pour les LLMs)
                splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=20)
            else:
                raise ValueError("Stratégie inconnue")
                
            chunks = splitter.split_documents(raw_docs)
            return raw_docs, chunks
    def generate_contextual_chunks(self, global_context: str, chunks: List[Document]) -> List[Document]:
        print(f"Génération du contexte pour {len(chunks)} chunks... (Cela va prendre ~2 minutes)")
        
        contextualized_docs = []
        
        for i, chunk in enumerate(chunks):
            # --- PAUSE DE SÉCURITÉ ANTI-429 ---
            if i > 0: 
                time.sleep(3.1)  # On attend 3.1 secondes entre chaque chunk
            # ----------------------------------

            prompt = (
                f"<document_context>{global_context}</document_context>\n"
                f"<chunk>{chunk.page_content}</chunk>\n"
                "Please give a short concise context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context."
            )
            
            try:
                response = self.llm.invoke(prompt)
                context = response.content
                
                # On crée un nouveau document avec le contexte ajouté
                new_doc = Document(
                    page_content=f"{context}\n\n{chunk.page_content}",
                    metadata=chunk.metadata.copy()
                )
                # Important : On injecte l'ID ici pour FAISS
                new_doc.metadata["chunk_id"] = i 
                
                contextualized_docs.append(new_doc)
                print(f"Traité {i+1}/{len(chunks)} chunks...")
                
            except Exception as e:
                print(f"⚠️ Erreur sur chunk {i}: {e}")
                # En cas d'erreur, on garde le chunk original pour ne pas le perdre
                chunk.metadata["chunk_id"] = i
                contextualized_docs.append(chunk)

        return contextualized_docs