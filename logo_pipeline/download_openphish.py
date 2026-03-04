#!/usr/bin/env python3
"""
download_openphish.py

Download OpenPhish feed, clean, deduplicate and append to an existing urls CSV.

Outputs:
 - data/openphish.txt          (raw feed)
 - data/openphish_clean.csv    (cleaned single-column CSV with header url)
 - data/urls_labels_expanded.csv (appended/merged file with label=phish)

Usage:
  python download_openphish.py

If you want to run it on a schedule, call this script from cron or GitHub Actions.
"""
import requests, os, csv, sys
from urllib.parse import urlparse
import pandas as pd

OPENPHISH_URL = "https://openphish.com/feed.txt"
OUT_DIR = "data"
RAW_PATH = os.path.join(OUT_DIR, "openphish.txt")
CLEAN_PATH = os.path.join(OUT_DIR, "openphish_clean.csv")
MERGED_PATH = os.path.join(OUT_DIR, "urls_labels_expanded.csv")  # existing aggregated file

def normalize_url(u):
    if not isinstance(u, str):
        return None
    u = u.strip()
    if not u:
        return None
    # ensure scheme
    if not u.startswith("http://") and not u.startswith("https://"):
        u = "http://" + u
    try:
        p = urlparse(u)
        # build canonical form: scheme://netloc/path?query (strip fragments)
        canonical = p.scheme.lower() + "://" + p.netloc.lower() + (p.path.rstrip('/') if p.path else '')
        if p.query:
            canonical += "?" + p.query
        return canonical
    except Exception:
        return None

def download_feed(url, out_raw):
    print("Downloading OpenPhish feed...")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    os.makedirs(os.path.dirname(out_raw), exist_ok=True)
    with open(out_raw, "w", encoding="utf-8") as f:
        f.write(r.text)
    print("Saved raw feed to", out_raw)
    return out_raw

def clean_and_save(raw_path, clean_csv_path):
    print("Cleaning feed and writing CSV:", clean_csv_path)
    urls = []
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            norm = normalize_url(line)
            if norm:
                urls.append(norm)
    # dedupe while preserving order
    seen = set()
    out_rows = []
    for u in urls:
        if u in seen: continue
        seen.add(u)
        out_rows.append({"url": u})
    # write CSV
    with open(clean_csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url"])
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)
    print("Saved", len(out_rows), "unique URLs to", clean_csv_path)
    return clean_csv_path

def merge_into_master(clean_csv, master_csv):
    # Load clean feed
    df_new = pd.read_csv(clean_csv)
    df_new["label"] = "phish"
    df_new["source"] = "openphish"
    # If master exists, read and append then dedupe by url (master label kept if already present)
    if os.path.exists(master_csv):
        print("Merging into existing master:", master_csv)
        df_master = pd.read_csv(master_csv)
        # Ensure master has url,label,source columns
        if "url" not in df_master.columns:
            print("ERROR: master file exists but has no 'url' column:", master_csv)
            return
        # mark existing urls to keep their labels (prefer existing labels)
        df_master = df_master[["url"] + [c for c in df_master.columns if c!="url"]]
        # concat and drop duplicates keeping first (master first so existing label preserved)
        df_combined = pd.concat([df_master, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
        df_combined.to_csv(master_csv, index=False)
        print("Master updated. Total rows now:", len(df_combined))
    else:
        print("Master file not found. Creating new master at:", master_csv)
        df_new.to_csv(master_csv, index=False)
        print("Master created. Total rows:", len(df_new))

def main():
    try:
        raw = download_feed(OPENPHISH_URL, RAW_PATH)
        clean = clean_and_save(raw, CLEAN_PATH)
        merge_into_master(clean, MERGED_PATH)
        print("OpenPhish import complete.")
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
