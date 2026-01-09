#!/usr/bin/env python3
"""
rag_pipeline.py

Single-file RAG pipeline for:
 - recipe FAISS index + recipes_clean.json
 - bone FAISS index + fish_bone_docs.json

Behavior:
 - Embed user query (with retry logic for Quotas)
 - Search recipe & bone FAISS indices
 - Pull JSON rows
 - Build constrained prompt
 - Call Gemini (using new google-genai SDK)
 - Return validated JSON
"""

import os
import json
import logging
import argparse
import time
from dotenv import load_dotenv
from typing import List, Dict, Any

import faiss
import numpy as np
from rapidfuzz import process

# NEW SDK IMPORT
from google import genai
from google.genai import types

load_dotenv()

# -----------------------
# Config / constants
# -----------------------
# Switched to the newer, stable embedding model
EMBED_MODEL = "text-embedding-004"
GEN_MODEL_NAME = "gemini-2.5-flash"

# Model output schema required fields
RECIPE_REQUIRED_FIELDS = {
    "recipe_id",
    "recipe_name",
    "cook_time",
    "ingredients",
    "steps",
    "tips",
    "youtube",
    "health_warning"
}

# logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("rag")

# -----------------------
# Setup
# -----------------------
def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        # Fallback for hackathon testing if env var is missing
        # key = "YOUR_HARDCODED_KEY_IF_NEEDED"
        raise RuntimeError("GEMINI_API_KEY environment variable not set.")
    
    return genai.Client(api_key=key)

# Initialize Client
client = get_client()

# -----------------------
# Utilities
# -----------------------
def embed(text: str, retries: int = 5) -> np.ndarray:
    """
    Return L2-normalized embedding as float32 1D array.
    Includes RETRY logic for 429 Quota Exceeded errors.
    """
    base_delay = 10  # Start with 10 seconds
    
    for attempt in range(retries):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
            # New SDK returns an object, we need the values
            emb = resp.embeddings[0].values
            arr = np.array(emb, dtype="float32")
            faiss.normalize_L2(arr.reshape(1, -1))
            return arr

        except Exception as e:
            # Check for Resource Exhausted / 429 errors
            error_str = str(e)
            if "429" in error_str or "ResourceExhausted" in error_str:
                wait_time = base_delay * (attempt + 1)
                logger.warning(f"Quota exceeded (Attempt {attempt+1}/{retries}). Sleeping {wait_time}s...")
                time.sleep(wait_time)
            else:
                # If it's a different error, raise it immediately
                logger.error(f"Embedding failed: {e}")
                raise e
    
    raise RuntimeError(f"Failed to get embedding after {retries} retries.")

def load_index_and_docs(index_path: str, docs_path: str):
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not os.path.isfile(docs_path):
        raise FileNotFoundError(f"Docs JSON not found: {docs_path}")

    index = faiss.read_index(index_path)
    with open(docs_path, "r", encoding="utf-8") as fh:
        docs = json.load(fh)
    
    # Optional: Warn if mismatch, but don't crash (useful during hackathons)
    if index.ntotal != len(docs):
        logger.warning(
            f"MISMATCH WARNING: FAISS index ({index.ntotal}) vs Docs ({len(docs)}). "
            "Ensure files are synced."
        )
    return index, docs

def fuzzy_match_fish(name: str, candidates: List[str]) -> str:
    if not candidates:
        return name
    match, score, _ = process.extractOne(name, candidates)
    logger.info("Fuzzy match: '%s' -> '%s' (score %.1f)", name, match, score)
    return match

# -----------------------
# RAG pieces
# -----------------------
def search_index(index: faiss.Index, query: str, top_k: int = 5) -> List[int]:
    q_emb = embed(query)
    D, I = index.search(q_emb.reshape(1, -1), top_k)
    # I is shape (1, top_k) with row indices (int). -1 indicates missing results.
    hits = [int(i) for i in I[0] if int(i) >= 0]
    logger.info("Search for '%s' -> %d hits", query, len(hits))
    return hits

def bone_rag(fish_name: str, bone_index: faiss.Index, bone_docs: List[Dict[str, Any]]):
    hits = search_index(bone_index, f"{fish_name} bone handling cooking warning", top_k=1)
    if not hits:
        return {"title": "Bone & Handling Advisory", "text": ""}
    doc = bone_docs[hits[0]]
    
    # Handle list or string format in JSON
    warnings = doc.get("handling_and_cooking_warnings", "")
    text = " ".join(warnings) if isinstance(warnings, list) else warnings
    
    return {"title": "Bone & Handling Advisory", "text": text}

def recipe_rag(fish_name: str, location: str, recipe_index: faiss.Index, recipe_docs: List[Dict[str, Any]], top_k: int = 12):
    hits = search_index(recipe_index, f"{fish_name} {location} traditional fish recipe", top_k=top_k)
    candidates = [recipe_docs[i] for i in hits]
    
    # Prioritize same state + popularity_score
    ranked = sorted(
        candidates,
        key=lambda r: (
            r.get("state") == location,
            r.get("popularity_score", 0)
        ),
        reverse=True
    )
    return ranked[:7]

# -----------------------
# Prompt builder + model call
# -----------------------
def build_prompt(fish: str, location: str, spice: str, bone_text: str, recipe_refs: List[Dict[str, Any]]) -> str:
    ref_block = ""
    for i, r in enumerate(recipe_refs, 1):
        recipe_name = r.get("recipe", {}).get("english") or r.get("recipe_name") or r.get("name", "Unknown")
        origin = r.get("state", "Unknown")
        outline = ", ".join(r.get("steps_outline", [])) if isinstance(r.get("steps_outline", []), list) else r.get("steps_outline", "")
        ref_block += f"""
REFERENCE {i}
Recipe Name: {recipe_name}
Origin: {origin}
Source: Traditional regional cooking
Outline: {outline}
"""
    prompt = f"""
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
CONTEXT
--------------------------------------------------
Fish: {fish}
Location: {location}
Spice preference: {spice}

Verified bone & handling information (already handled separately):
{bone_text}

--------------------------------------------------
RECIPE REFERENCES
--------------------------------------------------
{ref_block}

--------------------------------------------------
TASK
--------------------------------------------------
From the references above, generate 5 to 7 complete recipes.

For EACH recipe, include:
recipe_id, recipe_name, cook_time, ingredients, steps, tips, youtube, health_warning
"""
    return prompt

def call_model_and_get_json(prompt: str):
    logger.info("Sending prompt to Gemini (len=%d chars)", len(prompt))
    
    try:
        # Use JSON mode in new SDK
        response = client.models.generate_content(
            model=GEN_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.3
            )
        )
        
        # New SDK response parsing
        if not response.text:
            raise ValueError("Empty response from model")
            
        data = json.loads(response.text)
        return data

    except Exception as e:
        logger.error("Model generation failed.")
        # If possible, print response for debugging
        if 'response' in locals() and response.text:
            logger.error(response.text[:500])
        raise RuntimeError("Model output not valid JSON or API error") from e

# -----------------------
# Output validation
# -----------------------
def validate_model_output(out: Any):
    if not isinstance(out, list):
        # Sometimes model wraps it in {"recipes": [...]}, handle that gracefully
        if isinstance(out, dict) and "recipes" in out:
            return validate_model_output(out["recipes"])
        raise AssertionError("Model output is not a JSON list/array.")
        
    if not (1 <= len(out) <= 10):
        logger.warning("Model returned %d recipes (expected 5-7).", len(out))
    
    # check required fields for each recipe
    for i, r in enumerate(out):
        if not isinstance(r, dict):
            raise AssertionError(f"Recipe at index {i} is not an object.")
        missing = RECIPE_REQUIRED_FIELDS - set(r.keys())
        if missing:
            logger.warning(f"Recipe {i} missing fields: {missing}")

# -----------------------
# Main run function
# -----------------------
def run_rag(
    fish_name: str,
    location: str,
    spice_level: str,
    recipe_index_path: str,
    recipe_docs_path: str,
    bone_index_path: str,
    bone_docs_path: str
) -> Dict[str, Any]:
    
    recipe_index, recipe_docs = load_index_and_docs(recipe_index_path, recipe_docs_path)
    bone_index, bone_docs = load_index_and_docs(bone_index_path, bone_docs_path)

    # fuzzy match fish
    fish_candidates = [d.get("fish_common_name", "") for d in bone_docs if "fish_common_name" in d]
    fish = fuzzy_match_fish(fish_name, fish_candidates)

    bone_info = bone_rag(fish, bone_index, bone_docs)
    recipes_refs = recipe_rag(fish, location, recipe_index, recipe_docs)

    prompt = build_prompt(fish, location, spice_level, bone_info["text"], recipes_refs)
    model_out = call_model_and_get_json(prompt)
    
    # Handle case where model returns dict wrapper
    final_recipes = model_out["recipes"] if isinstance(model_out, dict) and "recipes" in model_out else model_out
    
    validate_model_output(final_recipes)

    return {
        "fish": fish,
        "location": location,
        "bone_warning": bone_info,
        "recipes": final_recipes
    }

# -----------------------
# CLI / smoke test
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline for fish recipes.")
    parser.add_argument("--fish", required=True)
    parser.add_argument("--location", required=True)
    parser.add_argument("--spice", default="medium")
    parser.add_argument("--recipe_index", required=True)
    parser.add_argument("--recipe_docs", required=True)
    parser.add_argument("--bone_index", required=True)
    parser.add_argument("--bone_docs", required=True)
    args = parser.parse_args()

    try:
        result = run_rag(
            fish_name=args.fish,
            location=args.location,
            spice_level=args.spice,
            recipe_index_path=args.recipe_index,
            recipe_docs_path=args.recipe_docs,
            bone_index_path=args.bone_index,
            bone_docs_path=args.bone_docs
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        logger.info("RAG completed successfully.")
    except Exception as e:
        logger.exception("RAG pipeline failed: %s", e)
        raise

if __name__ == "__main__":
    main()