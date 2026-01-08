import json
import numpy as np
import faiss
import ollama
import os

# -----------------------------------------------------------
# CONFIG (ONEDRIVE PATHS)
# -----------------------------------------------------------

INPUT_JSON = r"C:\Users\nithi\OneDrive\Desktop\Fish_Recipe_Metadata.json"
OUTPUT_DIR = r"C:\Users\nithi\OneDrive\Desktop\recipe_vector_store"
EMBED_MODEL = "nomic-embed-text"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(OUTPUT_DIR, "faiss.index")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "embeddings.npy")
CLEAN_JSON_PATH = os.path.join(OUTPUT_DIR, "recipes_clean.json")

# -----------------------------------------------------------
# LOAD RECIPE DATA
# -----------------------------------------------------------

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    recipes = json.load(f)

print(f"✅ Loaded {len(recipes)} recipes")

# -----------------------------------------------------------
# SPICE NORMALIZATION (CRITICAL FIX)
# -----------------------------------------------------------

def spice_label(value: str) -> str:
    mapping = {
        "1": "mild",
        "2": "medium",
        "3": "medium high",
        "4": "high",
        "5": "very high"
    }
    return mapping.get(str(value), "medium")

# -----------------------------------------------------------
# EMBEDDING FUNCTION (OLLAMA)
# -----------------------------------------------------------

def get_embedding(text: str):
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )
    return response["embedding"]

# -----------------------------------------------------------
# PREPARE TEXT FOR EMBEDDING (THIS DEFINES QUALITY)
# -----------------------------------------------------------

texts = []

for r in recipes:
    spice_text = spice_label(r["spicy_level"])

    text = (
        f"Recipe name: {r['recipe']['english']} also called {r['recipe']['local']}. "
        f"Traditional dish from {r['state']}, India. "
        f"Fish used: {' '.join(r['fish']['english'])}. "
        f"Spice level: {spice_text}. "
        f"Popularity score: {r['popularity_score']}. "
        f"Chef rating: {r['chef_rating']}. "
        f"Cooking type: fully cooked fish recipe."
    )

    texts.append(text)

# -----------------------------------------------------------
# GENERATE EMBEDDINGS
# -----------------------------------------------------------

print("🔄 Generating embeddings...")

embedding_list = []

for i, text in enumerate(texts):
    emb = get_embedding(text)
    embedding_list.append(emb)

    if (i + 1) % 20 == 0:
        print(f"  Embedded {i + 1}/{len(texts)} recipes")

embeddings = np.array(embedding_list, dtype="float32")

print("✅ Embeddings shape:", embeddings.shape)

# -----------------------------------------------------------
# BUILD FAISS INDEX (COSINE SIMILARITY)
# -----------------------------------------------------------

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print("✅ FAISS index built")
print("   Total vectors indexed:", index.ntotal)

# -----------------------------------------------------------
# SAVE OUTPUTS
# -----------------------------------------------------------

faiss.write_index(index, FAISS_INDEX_PATH)
np.save(EMBEDDINGS_PATH, embeddings)

with open(CLEAN_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(recipes, f, indent=2, ensure_ascii=False)

print("💾 Files saved successfully:")
print(" -", FAISS_INDEX_PATH)
print(" -", EMBEDDINGS_PATH)
print(" -", CLEAN_JSON_PATH)

print("🎉 RECIPE EMBEDDING PIPELINE COMPLETE")
