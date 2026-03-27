#!/usr/bin/env python3
# scripts/fetch_batch_resumable.py
"""
Wrapper that processes a batch CSV line-by-line and calls the existing
fetch_favicons_and_screenshots.py for each URL that hasn't been fetched yet.
Adds resume, retries, and polite delay.
"""

import csv
import os
import sys
import time
import argparse
import subprocess
from urllib.parse import urlparse

def domain_from_url(u: str) -> str:
    """Extract domain from URL."""
    try:
        p = urlparse(u if u.startswith("http") else "http://" + u)
        host = p.netloc
        # remove port if present
        host = host.split(":")[0]
        return host.lower()
    except Exception:
        return None

def ensure_temp_csv(tmp_csv_path: str, header: tuple = ("url",)) -> None:
    """Ensure temp CSV exists with header."""
    # write header if not exists
    if not os.path.exists(tmp_csv_path):
        with open(tmp_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def append_url_to_csv(tmp_csv_path: str, url: str) -> None:
    """Append URL to temp CSV."""
    with open(tmp_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([url])

def run_fetch_once(fetch_script: str, tmp_csv: str, outdir: str, screenshots: bool) -> tuple:
    """Run fetch script once."""
    cmd = [sys.executable, fetch_script, "--input", tmp_csv, "--outdir", outdir]
    if screenshots:
        cmd.append("--screenshots")
    # call the fetch script
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return res.returncode, res.stdout + "\n" + res.stderr
    except subprocess.TimeoutExpired as e:
        return 2, f"timeout: {e}"

def main() -> int:
    """Main function."""
    p = argparse.ArgumentParser()
    p.add_argument("--batch", required=True, help="path to batch csv (url,label)")
    p.add_argument("--outdir", required=True, help="output directory root for fetched sites")
    p.add_argument("--fetch-script", default="fetch_favicons_and_screenshots.py", help="path to existing fetch script")
    p.add_argument("--delay", type=float, default=1.0, help="seconds to sleep between each URL")
    p.add_argument("--retries", type=int, default=2, help="number of retries for a failed URL")
    p.add_argument("--screenshots", action="store_true", help="pass --screenshots to fetch script")
    p.add_argument("--tmp-dir", default="/tmp/logo_fetch_tmp", help="temp folder for single-url CSVs")
    args = p.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    total = 0
    skipped = 0
    fetched = 0
    failed = 0

    with open(args.batch, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            url = row.get("url") or row.get("URL") or next(iter(row.values()))
            if not url:
                continue
            domain = domain_from_url(url)
            if not domain:
                print(f"[{total}] bad-url -> {url}  (skipping)")
                failed += 1
                continue

            domain_dir = os.path.join(args.outdir, domain)
            screenshot_path = os.path.join(domain_dir, "screenshot.png")
            favicon_exists = False
            # if screenshot exists, assume that domain is already processed
            if os.path.exists(screenshot_path):
                skipped += 1
                print(f"[{total}] skip (exists): {domain}")
                continue

            # not fetched yet -> create a tiny CSV with this URL and call fetch script
            tmp_csv = os.path.join(args.tmp_dir, f"tmp_{domain.replace('/', '_')}.csv")
            # write header + single url (overwrite)
            with open(tmp_csv, "w", newline="", encoding="utf-8") as tf:
                writer = csv.writer(tf)
                writer.writerow(["url"])
                writer.writerow([url])

            attempt = 0
            ok = False
            while attempt <= args.retries and not ok:
                attempt += 1
                print(f"[{total}] fetching ({attempt}/{args.retries}) {domain} -> {url}")
                code, out = run_fetch_once(args.fetch_script, tmp_csv, args.outdir, args.screenshots)
                # basic heuristic success: directory created OR screenshot created
                if os.path.exists(os.path.join(args.outdir, domain)) and (
                    os.path.exists(os.path.join(args.outdir, domain, "screenshot.png"))
                    or os.path.exists(os.path.join(args.outdir, domain, "favicon.ico"))
                    or os.path.exists(os.path.join(args.outdir, domain, "favicon.png"))
                ):
                    ok = True
                    fetched += 1
                    print(f"[{total}] fetched OK: {domain}")
                else:
                    print(f"[{total}] fetch failed (code {code}). stdout/stderr snippet:\n{out.splitlines()[-5:]}")
                    if attempt <= args.retries:
                        backoff = 2 ** attempt
                        print(f"  -> retrying after {backoff}s")
                        time.sleep(backoff)
            if not ok:
                failed += 1
                print(f"[{total}] FAILED after {args.retries} retries: {domain}")

            # polite delay between requests
            time.sleep(args.delay)

    print("Done. total:", total, "skipped:", skipped, "fetched:", fetched, "failed:", failed)
    return 0

if __name__ == "__main__":
    sys.exit(main())