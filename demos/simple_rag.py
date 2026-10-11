"""
Minimal RAG demo using FAISS + sentence-transformers.
Uses a small in-memory knowledge base for reproducibility (no external API keys required for embedding).
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Tiny knowledge base
DOCS = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Deep learning uses neural networks with many layers to model complex patterns.",
    "RAG stands for Retrieval-Augmented Generation and combines search with language models.",
    "Feature engineering is the process of creating new input features from raw data to improve model performance.",
    "Cross-validation is a technique to assess how a model will generalize to an independent dataset.",
    "A confusion matrix summarizes the performance of a classification algorithm.",
    "Overfitting occurs when a model learns the training data too well and fails to generalize.",
    "Gradient descent is an optimization algorithm used to minimize a loss function.",
]

def build_index(docs, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return model, index, embeddings

def retrieve(query, model, index, docs, k=3):
    q_emb = model.encode([query], convert_to_numpy=True).astype(np.float32)
    distances, indices = index.search(q_emb, k)
    results = [(docs[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
    return results

if __name__ == "__main__":
    print("Building RAG index...")
    model, index, _ = build_index(DOCS)
    
    query = "What is retrieval augmented generation?"
    print(f"\nQuery: {query}\n")
    hits = retrieve(query, model, index, DOCS, k=3)
    
    print("Top retrieved documents:")
    for i, (doc, dist) in enumerate(hits, 1):
        print(f"{i}. (distance={dist:.4f}) {doc}")
    
    print("\n--- Simple grounded answer (template) ---")
    context = " ".join([d for d, _ in hits])
    print(f"Based on the retrieved context: {context[:300]}...")
    print("RAG is a technique that first retrieves relevant documents and then uses them to ground the generation of an LLM answer.")
