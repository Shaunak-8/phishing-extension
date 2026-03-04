# server_logo.py
# Simple Flask server to compute embedding for an uploaded image and find top-k similar logos.
# Usage: python server_logo.py
# Requires: flask, pillow, numpy, tensorflow

import json, os, io, numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Config
EMBEDDING_DIM = 1280   # MobileNetV2 pooling='avg' -> 1280
LOGO_DB_PATH = "logo_db_embeddings.json"
TOP_K = 5

app = Flask(__name__)

# Load precomputed logo DB (brand -> {'embedding': [...], 'count': N})
def load_logo_db(path=LOGO_DB_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Logo DB not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        db = json.load(f)
    brands = []
    embeddings = []
    for brand, info in db.items():
        emb = info.get("embedding")
        if emb and len(emb) > 0:
            brands.append(brand)
            embeddings.append(np.array(emb, dtype=np.float32))
    if len(embeddings) == 0:
        raise ValueError("No embeddings found in logo DB.")
    emb_matrix = np.vstack(embeddings)  # shape (N_brands, EMBEDDING_DIM)
    # normalize for cosine similarity
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms
    return brands, emb_matrix

# Build MobileNetV2 embedding model
def build_embedding_model():
    base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))
    return base

# Compute embedding from PIL image
def image_to_embedding(pil_img, model):
    img = pil_img.convert("RGB").resize((224,224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)  # MobileNetV2 preprocessing
    emb = model.predict(arr)
    emb = emb.flatten().astype(np.float32)
    # normalize
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb
    return emb / norm

# Cosine similarity search (emb_db normalized, query normalized)
def top_k_similar(query_emb, emb_db, brands, k=TOP_K):
    # emb_db shape (N, D); both normalized -> dot product is cosine
    scores = np.dot(emb_db, query_emb)
    idxs = np.argsort(-scores)[:k]
    results = []
    for i in idxs:
        results.append({"brand": brands[i], "score": float(round(float(scores[i]), 6))})
    return results

# Global load
print("Loading logo DB and embedding model (this may take a few seconds)...")
brands_list, emb_matrix = load_logo_db()
_emb_model = build_embedding_model()
print("Loaded logo DB with", len(brands_list), "brands.")

@app.route("/predict/logo", methods=["POST"])
def predict_logo():
    """
    Accepts:
      - multipart/form-data with 'image' file
      - OR JSON with 'image_base64' (optional)
    Returns JSON:
      { "matches": [{"brand": "...", "score": 0.92}, ...], "top_match": {...} }
    """
    try:
        if 'image' in request.files:
            f = request.files['image']
            img = Image.open(f.stream)
        else:
            # try JSON base64
            data = request.get_json(silent=True) or {}
            b64 = data.get("image_base64")
            if not b64:
                return jsonify({"error":"no image provided (send multipart form 'image' or JSON image_base64)"}), 400
            import base64
            img_bytes = base64.b64decode(b64)
            img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": "cannot read image: " + str(e)}), 400

    # compute embedding
    try:
        q_emb = image_to_embedding(img, _emb_model)
    except Exception as e:
        return jsonify({"error": "failed to compute embedding: " + str(e)}), 500

    # ensure query normalized
    qnorm = np.linalg.norm(q_emb)
    if qnorm == 0:
        return jsonify({"error":"zero embedding"}), 500
    q_emb = q_emb / qnorm

    results = top_k_similar(q_emb, emb_matrix, brands_list, k=TOP_K)
    top = results[0] if results else None
    return jsonify({"matches": results, "top_match": top})

if __name__ == "__main__":
    # Run dev server
    app.run(host="0.0.0.0", port=5001, debug=True)
