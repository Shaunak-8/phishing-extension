import os
import argparse
import pandas as pd
import requests
from urllib.parse import urlparse
from PIL import Image
from io import BytesIO
import asyncio

from playwright.async_api import async_playwright

async def fetch_screenshot(url, out_path):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.set_viewport_size({"width": 1280, "height": 800})
            
            await page.goto(url, timeout=30000, wait_until="networkidle")
            await page.screenshot(path=out_path, full_page=True)

            await browser.close()

        print(f"Saved screenshot: {out_path}")
        return True

    except Exception as e:
        print(f"Screenshot failed for {url}: {e}")
        return False


def fetch_favicon(url, out_dir):
    try:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        # Try common favicon paths
        paths = [
            "/favicon.ico",
            "/favicon.png",
            "/favicon.jpg"
        ]

        for p in paths:
            test_url = base + p
            try:
                r = requests.get(test_url, timeout=10)
                if r.status_code == 200 and len(r.content) > 0:
                    ext = p.split(".")[-1]
                    out_path = os.path.join(out_dir, f"favicon.{ext}")
                    with open(out_path, "wb") as f:
                        f.write(r.content)
                    print(f"Saved favicon: {out_path}")
                    return True
            except requests.RequestException:
                continue

        print(f"No favicon found for {url}")
        return False
    except Exception as e:
        print(f"Error fetching favicon: {e}")
        return False

async def main(args):
    df = pd.read_csv(args.input)
    urls = df['url'].dropna().tolist()

    for url in urls:
        print(f"Fetching: {url}")
        parsed = urlparse(url)
        domain = parsed.netloc
        out_dir = os.path.join(args.outdir, domain)
        os.makedirs(out_dir, exist_ok=True)

        # Fetch favicon
        fetch_favicon(url, out_dir)

        # Fetch screenshot (if flag enabled)
        if args.screenshots:
            screenshot_path = os.path.join(out_dir, "screenshot.png")
            await fetch_screenshot(url, screenshot_path)

    print("Done. Processed", len(urls), "URLs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--screenshots", action="store_true", help="Enable screenshot saving")
    args = parser.parse_args()

    asyncio.run(main(args))