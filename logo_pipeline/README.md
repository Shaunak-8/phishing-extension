Logo & URL dataset pipeline
=============================

This folder contains three helper scripts to expand your phishing dataset and build logo embeddings.

1) collect_urls.py
   - Combine local or remote URL feeds into a deduplicated CSV
   - Remote fetch for OpenPhish/PhishTank requires internet access and API endpoints

2) fetch_favicons_and_screenshots.py
   - Given a CSV of URLs, downloads favicons into data/raw/<host>/favicon.*
   - Optionally take page screenshots (requires pyppeteer/playwright and headless Chromium)

3) build_logo_embeddings.py
   - Loads logos from logos/<brand> directories and computes embeddings using MobileNetV2 (if TensorFlow available)
     otherwise uses a lightweight fallback embedding (resized pixels + mean/std).

Usage examples (run locally where you have internet and Chrome for screenshots):
  python collect_urls.py --input mylist.txt --output data/urls_labels_expanded.csv
  python fetch_favicons_and_screenshots.py --input data/urls_labels_expanded.csv --outdir data/raw --screenshots
  python build_logo_embeddings.py --logos data/raw --out logo_db_embeddings.json

Notes:
 - Screenshots and TF-based embeddings require more compute and dependencies.
 - The scripts are designed to run locally; this sandbox may not have internet or GPU, so some steps may fail here.
