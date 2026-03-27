#!/usr/bin/env python3
"""
server_logo_fixed.py

Flask server that:
 - loads precomputed logo embeddings (logo_db_embeddings.json)
 - computes embeddings for uploaded images (favicons/screenshots) using MobileNetV2
 - returns top-k logo matches at /predict/logo
 - provides a simple heuristic URL scorer + blacklist loader
 - provides /predict/combined that combines URL score and logo similarity
"""

import os
import io
import json
import base64
import traceback
import re
from urllib.parse import urlparse

from PIL import Image, UnidentifiedImageError
import numpy as np
import requests

from flask import Flask, request, jsonify

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# -----------------------
# Config
# -----------------------
BASE_DIR = os.path.dirname(__file__)
LOGO_DB_PATH = os.path.join(BASE_DIR, "logo_db_embeddings.json")
PHISHING_BLACKLIST_PATH = os.path.join(BASE_DIR, "data", "phishing_combined.txt")
TOP_K = 5
EMBEDDING_DIM = 1280  # MobileNetV2 pooling='avg' -> 1280

app = Flask(__name__)

# -----------------------
# Logo DB loading + model
# -----------------------
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
    emb_matrix = np.vstack(embeddings)
    # normalize rows
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms
    return brands, emb_matrix

def build_embedding_model():
    # MobileNetV2 with pooling='avg' returns 1280-d vectors
    base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224,224,3))
    return base

print("Loading logo DB and embedding model (this may take a few seconds)...")
brands_list, emb_matrix = load_logo_db()
_emb_model = build_embedding_model()
print("Loaded logo DB with", len(brands_list), "brands.")

# -----------------------
# Image helpers
# -----------------------
def try_open_image_from_bytes(bts):
    try:
        img = Image.open(io.BytesIO(bts))
        img.load()
        return img
    except UnidentifiedImageError:
        raise
    except Exception as e:
        raise UnidentifiedImageError(f"Failed to decode image: {e}")

def fetch_image_from_url(url, timeout=15):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        resp.raise_for_status()
        if len(resp.content) == 0:
            raise ValueError("Downloaded content is empty")
        return resp.content
    except Exception as e:
        raise RuntimeError(f"Failed to fetch image from URL: {e}")

def image_to_embedding(pil_img, model):
    img = pil_img.convert("RGB").resize((224,224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    emb = model.predict(arr)
    emb = emb.flatten().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm == 0:
        return emb, 0.0
    return emb / norm, float(norm)

def top_k_similar(query_emb, emb_db, brands, k=TOP_K):
    # assumption: emb_db rows are normalized
    scores = np.dot(emb_db, query_emb)
    idxs = np.argsort(-scores)[:k]
    results = []
    for i in idxs:
        results.append({"brand": brands[i], "score": float(round(float(scores[i]), 6))})
    return results

# -----------------------
# /predict/logo endpoint
# -----------------------
@app.route("/predict/logo", methods=["POST"])
def predict_logo():
    try:
        img = None
        # 1) multipart upload
        if 'image' in request.files and request.files['image'].filename:
            f = request.files['image']
            bts = f.read()
            try:
                img = try_open_image_from_bytes(bts)
            except Exception as e:
                return jsonify({"error": "uploaded file could not be decoded as an image", "detail": str(e)}), 400
        else:
            # 2) JSON: image_base64 or image_url
            data = request.get_json(silent=True) or {}
            if 'image_base64' in data:
                try:
                    b = base64.b64decode(data['image_base64'])
                    img = try_open_image_from_bytes(b)
                except Exception as e:
                    return jsonify({"error":"failed to decode base64 image", "detail": str(e)}), 400
            elif 'image_url' in data:
                try:
                    b = fetch_image_from_url(data['image_url'])
                    img = try_open_image_from_bytes(b)
                except Exception as e:
                    return jsonify({"error":"failed to fetch or decode image from URL", "detail": str(e)}), 400
            else:
                return jsonify({"error":"no image provided; send multipart 'image' or JSON 'image_base64'/'image_url'"}), 400

        q_emb, raw_norm = image_to_embedding(img, _emb_model)
        results = top_k_similar(q_emb, emb_matrix, brands_list, k=TOP_K)
        top = results[0] if results else None
        return jsonify({"matches": results, "top_match": top, "raw_embedding_norm": raw_norm})
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"error":"internal_server_error", "detail": str(e), "trace": tb}), 500

# -----------------------
# Simple heuristic URL scorer + blacklist loader
# -----------------------
_blacklist_domains = None
def _load_blacklist():
    global _blacklist_domains
    if _blacklist_domains is None:
        s = set()
        try:
            with open(PHISHING_BLACKLIST_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    u = line.strip()
                    if not u:
                        continue
                    if u.startswith("http://") or u.startswith("https://"):
                        try:
                            p = urlparse(u)
                            host = p.netloc.lower()
                            s.add(host)
                        except:
                            s.add(u.lower())
                    else:
                        s.add(u.lower())
        except FileNotFoundError:
            s = set()
        _blacklist_domains = s
    return _blacklist_domains

def _is_ip_host(host):
    return re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host) is not None

def _path_suspicious_tokens(path):
    if not path:
        return 0
    tokens = re.split(r"[\/_\-\.]+", path.lower())
    suspicious = {"login","verify","account","update","signin","bank","secure","payment","confirm","submit"}
    return int(any(t in suspicious for t in tokens))

def score_url_simple(url):
    """
    Heuristic phishing probability in [0,1] (higher => more suspicious)
    """
    if not url:
        return 0.5
    try:
        p = urlparse(url if url.startswith("http") else "http://" + url)
    except Exception:
        return 0.6
    host = (p.netloc or "").lower()
    path = p.path or ""
    query = p.query or ""

    # features
    is_https = 1 if p.scheme == "https" else 0
    blacklist = _load_blacklist()
    in_blacklist = 1 if host in blacklist else 0
    has_ip = 1 if _is_ip_host(host) else 0
    puny = 1 if host.startswith("xn--") else 0
    long_host = 1 if len(host) > 30 else 0
    many_dashes = 1 if host.count("-") >= 2 else 0
    path_susp = _path_suspicious_tokens(path + " " + query)

    # weights (tunable)
    score = 0.0
    score += in_blacklist * 0.9
    score += (1 - is_https) * 0.15
    score += has_ip * 0.25
    score += puny * 0.25
    score += long_host * 0.10
    score += many_dashes * 0.08
    score += path_susp * 0.20

    if score > 1.0:
        score = 1.0
    if score < 0.0:
        score = 0.0

    if score == 0:
        return 0.05
    return float(score)

# -----------------------
# Helper: compute logo result from current request (re-uses predict_logo logic)
# -----------------------
def compute_logo_from_request():
    """
    Reads image from Flask request (multipart or JSON base64) and returns (brand, score)
    """
    from flask import request as _r
    pil_img = None
    # try multipart
    if 'image' in _r.files and _r.files['image'].filename:
        try:
            pil_img = Image.open(_r.files['image'].stream)
        except Exception:
            pil_img = None
    else:
        data = _r.get_json(silent=True) or {}
        if 'image_base64' in data:
            try:
                b = base64.b64decode(data['image_base64'])
                pil_img = Image.open(io.BytesIO(b))
            except Exception:
                pil_img = None
        elif 'image_url' in data:
            try:
                b = fetch_image_from_url(data['image_url'])
                pil_img = Image.open(io.BytesIO(b))
            except Exception:
                pil_img = None

    if pil_img is None:
        return None, 0.0

    q_emb, _ = image_to_embedding(pil_img, _emb_model)
    results = top_k_similar(q_emb, emb_matrix, brands_list, k=1)
    if not results:
        return None, 0.0
    top = results[0]
    return top.get("brand"), float(top.get("score", 0.0))

# -----------------------
# Combined endpoint
# -----------------------
from flask import request as _req

# at top of file (if not already present)
from urllib.parse import urlparse

@app.route("/predict/combined", methods=["POST"])
def predict_combined():
    """
    Combined predictor endpoint.

    Accepts:
      - form field 'url' (optional)
      - multipart 'image' (optional) OR JSON image_base64 OR image_url

    Returns JSON:
      {
        "url": "...",
        "url_score": 0.5,
        "logo": {"brand": "...", "logo_score": 0.12},
        "combined_score": 0.62,
        "decision": "phish"|"legit"|"ambiguous",
        "reason": "optional string"
      }
    """
    # ------------ CONFIG ------------
    LOGO_MIN_CONFIDENCE = 0.40   # require at least this score to consider a logo match
    HOSTNAME_CONFIRM = False     # <--- set True to require hostname contains brand token (safer)
    # --------------------------------

    # read URL from form or JSON
    url = _req.form.get("url") or (_req.get_json(silent=True) or {}).get("url")
    url_score = score_url_simple(url) if url else 0.5

    # compute logo result (this reuses the file's compute_logo_from_request() helper)
    try:
        logo_brand, logo_score = compute_logo_from_request()
    except Exception as e:
        print("[server] warn: logo compute failed:", e)
        logo_brand, logo_score = None, 0.0

    # Validate logo confidence threshold
    if not logo_brand or float(logo_score) < LOGO_MIN_CONFIDENCE:
        logo_brand = None
        logo_score = 0.0
    else:
        # Optionally perform hostname -> brand confirmation
        if HOSTNAME_CONFIRM:
            try:
                if url:
                    hostname = (urlparse(url).hostname or "").lower()
                    brand_token = logo_brand.lower()
                    # If brand looks like a URL/domain, extract hostname
                    if brand_token.startswith("http://") or brand_token.startswith("https://"):
                        try:
                            brand_token = urlparse(brand_token).hostname or brand_token
                        except Exception:
                            pass
                    # reduce brand token to root if it contains dots
                    if "." in brand_token:
                        brand_token_root = brand_token.split(".")[0]
                    else:
                        brand_token_root = brand_token

                    if (brand_token in hostname) or (brand_token_root in hostname):
                        # hostname matches brand token - keep match
                        pass
                    else:
                        # hostname does not contain brand token -> reject logo match
                        logo_brand = None
                        logo_score = 0.0
                else:
                    # no url available - conservative: reject ambiguous brand
                    logo_brand = None
                    logo_score = 0.0
            except Exception as e:
                print("[server] warn: hostname-brand confirmation failed:", e)
                logo_brand = None
                logo_score = 0.0
        else:
            # HOSTNAME_CONFIRM disabled -> accept logo match as returned by model
            pass

    # Combine scores into a single suspiciousness measure.
    # Higher combined_score => more suspicious.
    # Keep same weighted formula as earlier.
    combined_score = 0.65 * float(url_score) + 0.35 * (1.0 - float(logo_score))

    # Decision thresholds (tweak if needed)
    if combined_score >= 0.75:
        decision = "phish"
    elif combined_score <= 0.35:
        decision = "legit"
    else:
        decision = "ambiguous"

    # Blacklist override (immediate phish)
    try:
        blacklist = _load_blacklist()
        if url and (urlparse(url).netloc or "").lower() in blacklist:
            return jsonify({
                "url": url,
                "url_score": 1.0,
                "logo": {"brand": logo_brand, "logo_score": float(logo_score)},
                "combined_score": 1.0,
                "decision": "phish",
                "reason": "blacklist_override"
            })
    except Exception:
        # ignore blacklist errors and continue
        pass

    # Return final response
    return jsonify({
        "url": url,
        "url_score": float(url_score),
        "logo": {"brand": logo_brand, "logo_score": float(logo_score)},
        "combined_score": float(round(combined_score, 4)),
        "decision": decision
    })

# ------------------ end predict_combined() ------------------
# -----------------------
# Run server (development)
# -----------------------
if __name__ == "__main__":
    # optional: enable CORS if extension fetches from service worker
    try:
        from flask_cors import CORS
        CORS(app)
    except Exception:
        pass

    # small health endpoint to confirm server is alive
    @app.route("/health", methods=["GET"])
    def _health():
        return jsonify({"ok": True, "status": "server running"}), 200

    # Start Flask dev server on port 5001 and listen on localhost + LAN IP
    # Note: for production, replace with a proper WSGI server
    app.run(host="0.0.0.0", port=5001, debug=True)