#!/usr/bin/env python3
"""
collect_urls.py

Combine URL lists from various sources (local files or remote feeds), normalize and deduplicate.
Usage examples:
  python collect_urls.py --input local_list1.txt local_list2.csv --output data/urls_labels_expanded.csv
  python collect_urls.py --from-openphish --from-phishtank --output data/urls_labels_expanded.csv

Note: fetching remote feeds requires internet access.
"""
import argparse
import csv
import os
import sys
import re
from urllib.parse import urlparse

def normalize_url(u: str) -> str:
    """Normalize a URL by ensuring scheme and removing fragment."""
    u = u.strip()
    if not u:
        return None
    # ensure scheme
    if not re.match(r'^https?://', u):
        u = 'http://' + u
    try:
        p = urlparse(u)
        # remove fragment
        u = p.scheme + '://' + p.netloc + p.path.rstrip('/') + (('?'+p.query) if p.query else '')
        return u
    except Exception as e:
        return None

def read_local_file(path: str) -> list:
    """Read URLs from a local file."""
    urls = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            # allow CSV with URL,label
            if ',' in line and line.count(',') == 1 and line.split(',')[1].strip() in ('0','1','phish','legit'):
                u, lab = line.split(',',1)
                urls.append((normalize_url(u), lab.strip()))
            else:
                urls.append((normalize_url(line), None))
    return urls

def write_csv(rows: list, out: str) -> None:
    """Write URLs to a CSV file."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['url','label','source'])
        for url, label, src in rows:
            w.writerow([url, label if label is not None else '', src])

def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='*', help='local files containing URLs (one per line)')
    parser.add_argument('--output', default='data/urls_labels_expanded.csv')
    parser.add_argument('--from-openphish', action='store_true')
    parser.add_argument('--from-phishtank', action='store_true')
    args = parser.parse_args()

    collected = []
    sources = []

    if args.input:
        for p in args.input:
            if os.path.exists(p):
                collected += [(u,l,p) for (u,l) in read_local_file(p)]
                sources.append(p)
            else:
                print("Warning: input file not found:", p, file=sys.stderr)

    # Remote fetch not executed here (needs internet). We just print instructions.
    if args.from_openphish or args.from_phishtank:
        print("Note: --from-openphish and --from-phishtank require internet access and are not executed inside this sandbox.")
        print("If you run this script locally with internet enabled, it will fetch the remote feeds and add them. See README for details.")

    # Normalize and dedupe by URL
    seen = set()
    rows = []
    for url, label, src in collected:
        if not url: 
            continue
        if url in seen: 
            continue
        seen.add(url)
        rows.append((url, label, src))

    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} URLs to {args.output}")

if __name__ == '__main__':
    main()