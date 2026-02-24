import time
import json
import requests
from sklearn.metrics.pairwise import cosine_similarity
from src.models import ModelFactory
import numpy as np

def run_benchmark():
    # Charger Ground Truth
    with open("data/ground_truth.json", "r") as f:
        qa_pairs = json.load(f)
    
    embeddings_model = ModelFactory.get_embeddings()
    latencies = []
    similarities = []
    
    print(f"Début du benchmark sur {len(qa_pairs)} questions...")
    
    for item in qa_pairs:
        start_time = time.time()
        
        # Appel API
        response = requests.post("http://localhost:8000/query", json={"q": item["question"], "k": 3})
        
        end_time = time.time()
        latencies.append(end_time - start_time)
        
        if response.status_code == 200:
            generated_answer = response.json()["answer"]
            true_answer = item["answer"]
            
            # Calcul Cosine Similarity
            vec_gen = embeddings_model.embed_query(generated_answer)
            vec_true = embeddings_model.embed_query(true_answer)
            
            sim = cosine_similarity([vec_gen], [vec_true])[0][0]
            similarities.append(sim)
            print(f"Q: {item['question'][:30]}... | Latency: {end_time-start_time:.2f}s | Sim: {sim:.4f}")
        else:
            print("Erreur API")

    print("\n--- RÉSULTATS ---")
    print(f"Latence Moyenne: {np.mean(latencies):.4f} secondes")
    print(f"Similarité Sémantique Moyenne: {np.mean(similarities):.4f}")

if __name__ == "__main__":
    # Assurez-vous que l'API tourne dans un autre terminal
    run_benchmark()