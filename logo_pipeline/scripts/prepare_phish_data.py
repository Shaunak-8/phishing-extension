import pandas as pd
import re
import os
from urllib.parse import urlparse
from typing import List, Optional

os.makedirs("data/processed", exist_ok=True)

def normalize_url(u: str) -> Optional[str]:
    """
    Normalize a URL by removing default ports and converting to lowercase.
    
    Args:
    u (str): The URL to normalize.
    
    Returns:
    str: The normalized URL or None if the input is empty.
    """
    u = str(u).strip()
    if not u:
        return None
    # if it's raw IP or missing scheme, keep host+path
    if not re.match(r"^https?://", u):
        # try to handle 'domain/path' lines
        u = "http://" + u
    try:
        p = urlparse(u)
        host = p.netloc.lower()
        # remove default ports
        host = re.sub(r":80$","", host)
        path = p.path or ""
        query = ("?" + p.query) if p.query else ""
        return p.scheme + "://" + host + path + query
    except Exception:
        return None

def extract_from_urlhaus(path: str) -> List[str]:
    """
    Extract URLs from a URLhaus CSV file.
    
    Args:
    path (str): The path to the URLhaus CSV file.
    
    Returns:
    list: A list of extracted URLs.
    """
    if not os.path.exists(path):
        print("URLhaus not found:", path)
        return []

    colnames = [
        "id",
        "dateadded",
        "url",
        "url_status",
        "last_online",
        "threat",
        "tags",
        "urlhaus_link",
        "reporter"
    ]

    try:
        df = pd.read_csv(
            path,
            comment="#",
            encoding="latin1",
            on_bad_lines="skip",
            engine="python",
            header=None,          # <-- Tell pandas: NO header in file
            names=colnames        # <-- Force correct column names
        )
    except Exception as e:
        print("Failed to read URLhaus:", e)
        return []

    # Drop rows that still have column-name-like garbage
    df = df[df["url"].str.startswith("http")]

    urls = df["url"].dropna().astype(str).tolist()
    print("Extracted", len(urls), "URLhaus URLs")
    return urls

def extract_from_custom_csv(path: str) -> List[str]:
    """
    Extract URLs from a custom CSV file.
    
    Args:
    path (str): The path to the custom CSV file.
    
    Returns:
    list: A list of extracted URLs.
    """
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, low_memory=False)
    col = None
    for c in ['url','URL','link','Link']:
        if c in df.columns:
            col = c; break
    if col:
        return df[col].dropna().astype(str).tolist()
    else:
        return df.iloc[:,0].dropna().astype(str).tolist()

def load_benign(path: str = "data/feeds/benign.txt") -> List[str]:
    """
    Load benign URLs from a text file.
    
    Args:
    path (str): The path to the text file.
    
    Returns:
    list: A list of loaded URLs.
    """
    if not os.path.exists(path):
        return []
    with open(path,"r",encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    # convert domain lines to full http form
    urls = []
    for d in lines:
        if re.match(r"^https?://", d):
            urls.append(d)
        else:
            urls.append("http://" + d)
    return urls

def main() -> None:
    """
    The main function.
    """
    print("Loading URLhaus...")
    phish_urls = extract_from_urlhaus("data/feeds/urlhaus_recent.csv")
    print("Found", len(phish_urls), "raw URLhaus entries")
    # optional: PhishTank & OpenPhish
    phishtank = extract_from_custom_csv("data/feeds/phishtank.csv")
    print("PhishTank entries:", len(phishtank))
    openphish = extract_from_custom_csv("data/feeds/openphish.csv")
    print("OpenPhish entries:", len(openphish))

    phish_urls += phishtank + openphish

    # normalize & dedupe
    norm_phish = set()
    for u in phish_urls:
        n = normalize_url(u)
        if n:
            norm_phish.add(n)
    print("Normalized phishing URLs:", len(norm_phish))

    benign = load_benign("data/feeds/benign.txt")
    norm_benign = set()
    for u in benign:
        n = normalize_url(u)
        if n:
            norm_benign.add(n)
    print("Normalized benign URLs:", len(norm_benign))

    # remove any overlap (if a domain in both, keep as phish)
    norm_benign = {u for u in norm_benign if u not in norm_phish}

    # produce labeled dataframe
    rows = []
    for u in norm_phish:
        rows.append((u,1))
    for u in norm_benign:
        rows.append((u,0))

    df = pd.DataFrame(rows, columns=["url","label"])
    df.to_csv("data/processed/urls_labels_raw.csv", index=False)
    print("Saved data/processed/urls_labels_raw.csv with", len(df), "rows")

if __name__ == "__main__":
    main()