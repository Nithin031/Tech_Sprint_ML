import os
import json
import faiss
import numpy as np
from rapidfuzz import process
import google.generativeai as genai

# =========================================================
# CONFIG (NO HARDCODED PATHS — PASSED AT RUNTIME)
# =========================================================

EMBED_MODEL = "models/embedding-001"
GEN_MODEL_NAME = "gemini-1.5-flash"

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEN_MODEL = genai.GenerativeModel(GEN_MODEL_NAME)

# =========================================================
# UTILS
# =========================================================

def embed(text: str) -> np.ndarray:
    emb = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query"
    )["embedding"]
    emb = np.array(emb, dtype="float32")
    faiss.normalize_L2(emb.reshape(1, -1))
    return emb

def load_index_and_docs(index_path, docs_path):
    return (
        faiss.read_index(index_path),
        json.load(open(docs_path, "r", encoding="utf-8"))
    )

def fuzzy_match_fish(name, candidates):
    match, _, _ = process.extractOne(name, candidates)
    return match

# =========================================================
# BONE RAG (TOP-1 ONLY)
# =========================================================

def bone_rag(fish_name, bone_index, bone_docs):
    q = embed(f"{fish_name} bone handling cooking warning")
    _, idx = bone_index.search(q.reshape(1, -1), 1)
    doc = bone_docs[idx[0][0]]

    return {
        "title": "Bone & Handling Advisory",
        "text": " ".join(doc.get("handling_and_cooking_warnings", []))
    }

# =========================================================
# RECIPE RAG (TOP 5–7)
# =========================================================

def recipe_rag(fish_name, location, recipe_index, recipe_docs):
    q = embed(f"{fish_name} {location} traditional fish recipe")
    _, idx = recipe_index.search(q.reshape(1, -1), 12)

    candidates = [recipe_docs[i] for i in idx[0]]

    # prioritize same state + popularity
    ranked = sorted(
        candidates,
        key=lambda r: (
            r.get("state") == location,
            r.get("popularity_score", 0)
        ),
        reverse=True
    )

    return ranked[:7]

# =========================================================
# PROMPT BUILDER (YOUR BIG PROMPT — LOCKED)
# =========================================================

def build_prompt(fish, location, spice, bone_text, recipe_refs):
    ref_block = ""
    for i, r in enumerate(recipe_refs, 1):
        ref_block += f"""
REFERENCE {i}
Recipe Name: {r['recipe']['english']}
Origin: {r['state']}
Source: Traditional regional cooking
Outline: {', '.join(r.get('steps_outline', []))}
"""

    return f"""
You are a professional home-cooking instructor and food writer.
Your job is to teach cooking in a way that a first-time cook can follow without confusion.

IMPORTANT CONSTRAINTS (DO NOT BREAK THESE):
1. You MUST use ONLY the recipe references provided below.
2. You MUST NOT invent any new recipes, variations, or ingredients.
3. You MUST NOT add bone-handling advice (this is already provided separately).
4. You MUST NOT add health warnings unless the dish is genuinely heavy, oily, or traditionally advised to be eaten in moderation.
5. You MUST give very detailed, step-by-step cooking instructions.
6. Every step must explain:
   - what to do
   - how long it takes (minutes/seconds)
   - heat level (low / medium / high)
   - what to look for to know it’s done
7. Assume the user is cooking at home with basic utensils.
8. The tone must feel human, calm, and reassuring — never robotic.

--------------------------------------------------
CONTEXT (DO NOT REPEAT THIS VERBATIM IN OUTPUT)
--------------------------------------------------
Fish: {fish}
Location: {location}
Spice preference: {spice}

Verified bone & handling information (already handled separately):
{bone_text}

--------------------------------------------------
RECIPE REFERENCES (TRUSTED SOURCES ONLY)
--------------------------------------------------
{ref_block}

--------------------------------------------------
TASK
--------------------------------------------------
From the references above, generate 5 to 7 complete recipes.

For EACH recipe, include:
recipe_id, recipe_name, cook_time, ingredients, steps, tips, youtube, health_warning

--------------------------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------------------------
Return ONLY a JSON array. No extra text.
"""

# =========================================================
# MAIN CALLABLE FUNCTION (THIS IS WHAT YOU USE)
# =========================================================

def run_rag(
    fish_name,
    location,
    spice_level,
    recipe_index_path,
    recipe_docs_path,
    bone_index_path,
    bone_docs_path
):
    # load indices
    recipe_index, recipe_docs = load_index_and_docs(recipe_index_path, recipe_docs_path)
    bone_index, bone_docs = load_index_and_docs(bone_index_path, bone_docs_path)

    # autocorrect fish name
    fish_candidates = [d["fish_common_name"] for d in bone_docs]
    fish = fuzzy_match_fish(fish_name, fish_candidates)

    # bone RAG
    bone_info = bone_rag(fish, bone_index, bone_docs)

    # recipe RAG
    recipes = recipe_rag(fish, location, recipe_index, recipe_docs)

    # prompt
    prompt = build_prompt(
        fish,
        location,
        spice_level,
        bone_info["text"],
        recipes
    )

    # Gemini call
    response = GEN_MODEL.generate_content(prompt)
    recipe_json = json.loads(response.text)

    return {
        "fish": fish,
        "location": location,
        "bone_warning": bone_info,
        "recipes": recipe_json
    }
