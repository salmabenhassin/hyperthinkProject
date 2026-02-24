# 📊 RAG Benchmark Report

**Date:** 2026-02-24 21:24:54

**Document:** Attention Is All You Need
**Evaluated Questions:** 10

## 📈 Global Summary

| Metric | Average Value |
| :--- | :--- |
| **Average Latency** | **4.5379 s** |
| **Semantic Similarity** | **0.6142** |

## 🧐 Detailed Analysis

| ID | Question | Latency (s) | Similarity | Status |
| :--- | :--- | :--- | :--- | :--- |
| 1 | How many identical layers does the encoder have? | 12.80 | 0.9052 | 🟢 Excellent |
| 2 | What is the dimension of d_model? | 3.36 | 0.5798 | 🔴 Poor |
| 3 | Which hardware was used for training the models? | 3.14 | 0.8247 | 🟢 Excellent |
| 4 | How long did the training of the big model take? | 2.84 | 0.7575 | 🟡 Good |
| 5 | What is the BLEU score of the big model on the EN-DE task? | 2.74 | 0.1661 | 🔴 Poor |
| 6 | What optimizer was used for training? | 3.46 | 0.9026 | 🟢 Excellent |
| 7 | What is the formula for the attention mechanism? | 5.00 | 0.7399 | 🟡 Good |
| 8 | Why did the authors choose sinusoidal positional encodings? | 4.42 | 0.5218 | 🔴 Poor |
| 9 | How is the decoder different from the encoder regarding attention? | 4.47 | 0.7095 | 🟡 Good |
| 10 | What regularization technique was applied to the sums of embeddings? | 3.15 | 0.0350 | 🔴 Poor |

### 🔍 Answer Comparison (Sample)

**✅ Best Match (Score: 0.9052)**
> **Q:** How many identical layers does the encoder have?
> **Expected:** The encoder is composed of a stack of N = 6 identical layers.
> **Generated:** The encoder has 6 identical layers.

**⚠️ Needs Improvement (Score: 0.0350)**
> **Q:** What regularization technique was applied to the sums of embeddings?
> **Expected:** In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.
> **Generated:** I cannot answer this based on the provided context.
