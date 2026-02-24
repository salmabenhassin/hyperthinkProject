import json
import time
import requests
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/query"
GROUND_TRUTH_PATH = "data/ground_truth.json"
REPORT_FILE = "BENCHMARK_REPORT.md"

# Modèle pour l'évaluation de similarité (Sémantique)
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_ground_truth():
    with open(GROUND_TRUTH_PATH, "r") as f:
        return json.load(f)

def run_benchmark():
    qa_pairs = load_ground_truth()
    results = []
    
    print(f"🚀 Démarrage du benchmark sur {len(qa_pairs)} questions...")

    total_latency = 0
    total_similarity = 0

    for i, item in enumerate(qa_pairs):
        question = item["question"]
        expected_answer = item["answer"]
        
        # 1. Mesure de la Latence
        start_time = time.time()
        try:
            response = requests.post(API_URL, json={"q": question, "k": 5})
            response.raise_for_status()
            data = response.json()
            generated_answer = data["answer"]
        except Exception as e:
            generated_answer = "Error"
            print(f"❌ Erreur sur Q{i+1}: {e}")
        
        end_time = time.time()
        latency = end_time - start_time
        
        # 2. Calcul de la Similarité Cosinus
        embeddings = model.encode([expected_answer, generated_answer])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Stockage
        results.append({
            "id": i + 1,
            "question": question,
            "latency": latency,
            "similarity": similarity,
            "generated": generated_answer,
            "expected": expected_answer
        })
        
        total_latency += latency
        total_similarity += similarity
        print(f"✅ Q{i+1} | Latence: {latency:.2f}s | Sim: {similarity:.4f}")

    # --- GÉNÉRATION DU RAPPORT MARKDOWN ---
    avg_latency = total_latency / len(qa_pairs)
    avg_similarity = total_similarity / len(qa_pairs)

    print(f"\n📝 Génération du rapport dans {REPORT_FILE}...")
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        # En-tête du rapport
        f.write(f"# 📊 RAG Benchmark Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Document:** Attention Is All You Need\n")
        f.write(f"**Evaluated Questions:** {len(qa_pairs)}\n\n")
        
        # Résumé Global
        f.write("## 📈 Global Summary\n\n")
        f.write("| Metric | Average Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Average Latency** | **{avg_latency:.4f} s** |\n")
        f.write(f"| **Semantic Similarity** | **{avg_similarity:.4f}** |\n\n")
        
        # Détails par question
        f.write("## 🧐 Detailed Analysis\n\n")
        f.write("| ID | Question | Latency (s) | Similarity | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        
        for res in results:
            # Petite icône pour visualiser la qualité
            status = "🟢 Excellent" if res["similarity"] > 0.8 else "🟡 Good" if res["similarity"] > 0.6 else "🔴 Poor"
            f.write(f"| {res['id']} | {res['question']} | {res['latency']:.2f} | {res['similarity']:.4f} | {status} |\n")

        # Section Comparaison (Optionnelle pour le détail)
        f.write("\n### 🔍 Answer Comparison (Sample)\n\n")
        best_res = max(results, key=lambda x: x['similarity'])
        worst_res = min(results, key=lambda x: x['similarity'])
        
        f.write(f"**✅ Best Match (Score: {best_res['similarity']:.4f})**\n")
        f.write(f"> **Q:** {best_res['question']}\n")
        f.write(f"> **Expected:** {best_res['expected']}\n")
        f.write(f"> **Generated:** {best_res['generated']}\n\n")

        f.write(f"**⚠️ Needs Improvement (Score: {worst_res['similarity']:.4f})**\n")
        f.write(f"> **Q:** {worst_res['question']}\n")
        f.write(f"> **Expected:** {worst_res['expected']}\n")
        f.write(f"> **Generated:** {worst_res['generated']}\n")

    print(f"✨ Rapport généré avec succès !")

if __name__ == "__main__":
    run_benchmark()