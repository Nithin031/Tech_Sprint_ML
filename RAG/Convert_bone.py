import json
import os
import numpy as np
import faiss
import ollama

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

INPUT_JSON = r"D:\Hackathon\Tech Sprint\RAG\fish_bone_knowledge.json"
OUTPUT_DIR = r"D:\Hackathon\Tech Sprint\RAG"

FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "fish_bone_faiss.index")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "fish_bone_embeddings.npy")
DOCS_PATH = os.path.join(OUTPUT_DIR, "fish_bone_docs.json")

EMBED_MODEL = "nomic-embed-text"

# -----------------------------------------------------------
# LOAD BONE KNOWLEDGE
# -----------------------------------------------------------

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    bone_docs = json.load(f)

print(f"✅ Loaded {len(bone_docs)} bone knowledge records")

# -----------------------------------------------------------
# PREPARE TEXT FOR EMBEDDING
# -----------------------------------------------------------

texts = []

for b in bone_docs:
    text = (
        f"Fish: {b['fish_common_name']}. "
        f"Scientific name: {b['scientific_name']}. "
        f"Regional names: {', '.join(b['regional_names'])}. "
        f"Bone complexity: {b['bone_profile']['complexity']}. "
        f"Handling and cooking warnings: {', '.join(b['handling_and_cooking_warnings'])}. "
        f"Source: {b['source']}."
    )
    texts.append(text)

# -----------------------------------------------------------
# GENERATE EMBEDDINGS (OLLAMA)
# -----------------------------------------------------------

print("🔄 Generating bone embeddings using Ollama...")

embeddings = []

for i, text in enumerate(texts):
    emb = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )["embedding"]

    embeddings.append(emb)

    if (i + 1) % 5 == 0:
        print(f"  Embedded {i + 1}/{len(texts)} records")

embeddings = np.array(embeddings, dtype="float32")

print("✅ Embeddings shape:", embeddings.shape)

# -----------------------------------------------------------
# BUILD FAISS INDEX (COSINE SIMILARITY)
# -----------------------------------------------------------

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, FAISS_INDEX_PATH)
np.save(EMBEDDINGS_PATH, embeddings)

# Save docs for retrieval
with open(DOCS_PATH, "w", encoding="utf-8") as f:
    json.dump(bone_docs, f, indent=2, ensure_ascii=False)

print("🎉 Bone knowledge FAISS index created (local)")
print("📁 Files saved:")
print(" -", FAISS_INDEX_PATH)
print(" -", EMBEDDINGS_PATH)
print(" -", DOCS_PATH)
