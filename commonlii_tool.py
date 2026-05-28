import os
import re
import time
import requests
from urllib.parse import urljoin, quote

BASE_URL      = "https://www.judiciary.gov.sg"
JUDGMENTS_URL = "https://www.judiciary.gov.sg/judgments/judgments-case-summaries"
DOWNLOAD_DIR  = "data/cases/criminal"
REQUEST_DELAY = 2
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.5",
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip().replace(" ", "_")
    return name[:100] + ".txt"

def _strip_html(html: str) -> str:
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<p[^>]*>', '\n', text)
    text = re.sub(r'</p>', '\n', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def _fetch(url: str) -> str:
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"[JUDICIARY] Fetch failed: {e}")
        return ""

def _save_case(title: str, text: str, url: str, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    filename = _safe_filename(title)
    path     = os.path.join(save_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"SOURCE: Singapore Courts (judiciary.gov.sg)\n")
        f.write(f"TITLE: {title}\n")
        f.write(f"URL: {url}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)
    return path

# ── SEARCH ────────────────────────────────────────────────────────────────────

def search(query: str, max_results: int = 10) -> list:
    """
    Search the Singapore Courts judgments listing.
    For reliable full-text search, prefer commonlii_tool.py.
    This tool covers only judgments listed on the judiciary.gov.sg front page.
    """
    print(f"[JUDICIARY] Fetching recent judgments listing...")
    html = _fetch(JUDGMENTS_URL)
    if not html:
        return []

    results     = []
    seen        = set()
    query_lower = query.lower()
    query_words = [w for w in re.split(r'\W+', query_lower) if len(w) >= 3]

    patterns = [
        r'href="(https://www\.judiciary\.gov\.sg/judgments/[^"]+)"[^>]*>\s*([^<]{5,200})</a>',
        r'href="(/judgments/[^"]+)"[^>]*>\s*([^<]{5,200})</a>',
    ]

    for pattern in patterns:
        for path, title in re.findall(pattern, html, re.IGNORECASE):
            title = title.strip()
            if not title:
                continue
            url = path if path.startswith("http") else BASE_URL + path
            if url in seen:
                continue
            seen.add(url)
            title_lower = title.lower()
            if any(w in title_lower for w in query_words):
                results.append({"title": title, "url": url})
                if len(results) >= max_results:
                    break
        if len(results) >= max_results:
            break

    print(f"[JUDICIARY] Found {len(results)} matching result(s)")
    if not results:
        print("[JUDICIARY] Tip: Use commonlii_tool.py for full-text search.")
    return results

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────

def download(results: list, max_download: int = 5, save_dir: str = DOWNLOAD_DIR) -> list:
    saved = []
    for i, result in enumerate(results[:max_download]):
        title = result["title"]
        url   = result["url"]
        print(f"\n[JUDICIARY] Downloading {i+1}/{min(len(results), max_download)}: {title}")
        html = _fetch(url)
        if not html:
            continue
        text = _strip_html(html)
        if len(text) < 300:
            continue
        path = _save_case(title, text, url, save_dir)
        print(f"  [SAVED] {path}")
        saved.append(path)
    return saved

def search_and_download(
    query: str,
    max_results: int = 10,
    max_download: int = 5,
    save_dir: str = DOWNLOAD_DIR
) -> list:
    """Combined search + download. For comprehensive search, prefer commonlii_tool.py."""
    results = search(query, max_results=max_results)
    if not results:
        return []
    return download(results, max_download=max_download, save_dir=save_dir)


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "drug trafficking"
    paths = search_and_download(query, max_results=10, max_download=3)
    if paths:
        print(f"\n[DONE] {len(paths)} judgments saved to {DOWNLOAD_DIR}")
        print("Run: python ingest.py")