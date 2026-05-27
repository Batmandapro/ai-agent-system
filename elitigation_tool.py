import os
import re
import time
import requests
from urllib.parse import urljoin, quote

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL     = "https://www.elitigation.sg"
SEARCH_URL   = "https://www.elitigation.sg/gd/Home/Index"
DOWNLOAD_DIR = "data/cases"
REQUEST_DELAY = 2   # seconds between requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Legal Research Assistant; educational use)"
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    """Convert a case name into a safe filename."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip().replace(" ", "_")
    return name[:100] + ".txt"

def _strip_html(html: str) -> str:
    """Strip HTML tags and clean up whitespace."""
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'\s{3,}', '\n\n', text)
    return text.strip()

def _search_elitigation(query: str, max_results: int = 10) -> list:
    """
    Search eLitigation for Singapore judgments matching the query.
    Returns a list of dicts with title and url.
    """
    params = {
        "Filter":    query,
        "YearOfDecision": "",
        "CourtType": ""
    }

    try:
        resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ELITIGATION] Search failed: {e}")
        return []

    html    = resp.text
    results = []
    seen    = set()

    # Parse judgment links
    pattern = r'href="(/gd/s/[^"]+)"[^>]*>\s*([^<]{5,})</a>'
    matches  = re.findall(pattern, html, re.IGNORECASE)

    for path, title in matches:
        url   = BASE_URL + path
        title = title.strip()
        if url in seen or not title:
            continue
        seen.add(url)
        results.append({"title": title, "url": url})
        if len(results) >= max_results:
            break

    return results

def _fetch_judgment_text(url: str) -> str:
    """Fetch the full text of a judgment from its eLitigation URL."""
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return _strip_html(resp.text)
    except Exception as e:
        print(f"[ELITIGATION] Fetch failed for {url}: {e}")
        return ""

def _save_case(title: str, text: str, url: str) -> str:
    """Save judgment text to data/cases/ as a .txt file."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    filename = _safe_filename(title)
    path     = os.path.join(DOWNLOAD_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"SOURCE: eLitigation (judiciary.gov.sg)\n")
        f.write(f"TITLE: {title}\n")
        f.write(f"URL: {url}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)

    return path

# ── PUBLIC API ────────────────────────────────────────────────────────────────

def search(query: str, max_results: int = 10) -> list:
    """
    Search eLitigation for Singapore judgments.

    Returns list of dicts:
    [{"title": "...", "url": "..."}, ...]
    """
    print(f"[ELITIGATION] Searching: {query}")
    results = _search_elitigation(query, max_results=max_results)
    print(f"[ELITIGATION] Found {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['title']}")
        print(f"     {r['url']}")
    return results

def download(results: list, max_download: int = 5) -> list:
    """
    Download and save judgment texts from search results.

    Args:
        results:      List from search()
        max_download: Maximum number of judgments to download

    Returns:
        List of saved file paths
    """
    saved = []
    for i, result in enumerate(results[:max_download]):
        title = result["title"]
        url   = result["url"]
        print(f"\n[ELITIGATION] Downloading {i+1}/{min(len(results), max_download)}: {title}")

        text = _fetch_judgment_text(url)
        if not text or len(text) < 200:
            print(f"  [SKIP] Insufficient text retrieved")
            continue

        path = _save_case(title, text, url)
        print(f"  [SAVED] {path} ({len(text)} chars)")
        saved.append(path)

    return saved

def search_and_download(query: str, max_results: int = 10, max_download: int = 5) -> list:
    """
    Combined search + download in one call.
    Returns list of saved file paths ready for ingest.py.
    """
    results = search(query, max_results=max_results)
    if not results:
        print("[ELITIGATION] No results found.")
        return []
    return download(results, max_download=max_download)


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "drug trafficking Misuse of Drugs Act"
    paths = search_and_download(query, max_results=10, max_download=3)
    if paths:
        print(f"\n[DONE] {len(paths)} judgments saved to data/cases/")
        print("Run python ingest.py to embed them into your database.")