import os
import re
import time
import requests
from urllib.parse import quote

# ── CONFIG ────────────────────────────────────────────────────────────────────
COMMONLII_BASE    = "http://www.commonlii.org"
SEARCH_URL        = "http://www.commonlii.org/cgi-bin/sinosrch.cgi"
DOWNLOAD_DIR      = "data/cases"
REQUEST_DELAY     = 2   # seconds between requests — be polite to the server
HEADERS           = {
    "User-Agent": "Mozilla/5.0 (Legal Research Assistant; educational use)"
}

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    """Convert a case name into a safe filename."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip().replace(" ", "_")
    return name[:100] + ".pdf" if not name.endswith(".pdf") else name[:100]

def _search_commonlii(query: str, jurisdiction: str = "sg", max_results: int = 10) -> list:
    """
    Search CommonLII for Singapore cases matching the query.
    Returns a list of dicts with title, url, and snippet.
    """
    params = {
        "query":  query,
        "method": "auto",
        "rank":   "on",
        "meta":   "/sg/",
        "results": str(max_results),
        "submit":  "Search"
    }

    try:
        resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"[COMMONLII] Search failed: {e}")
        return []

    html = resp.text
    results = []

    # Parse result links and titles from CommonLII HTML
    pattern = r'<a href="(/sg/cases/[^"]+\.html?)"[^>]*>([^<]+)</a>'
    matches  = re.findall(pattern, html, re.IGNORECASE)

    seen = set()
    for path, title in matches:
        url = COMMONLII_BASE + path
        if url in seen:
            continue
        seen.add(url)
        results.append({
            "title": title.strip(),
            "url":   url
        })
        if len(results) >= max_results:
            break

    return results

def _fetch_case_text(url: str) -> str:
    """Fetch the full text of a case from its CommonLII URL."""
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        html = resp.text

        # Strip HTML tags to get plain text
        text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&[a-z]+;', '', text)
        text = re.sub(r'\s{2,}', '\n', text)
        return text.strip()
    except Exception as e:
        print(f"[COMMONLII] Fetch failed for {url}: {e}")
        return ""

def _save_as_txt(title: str, text: str) -> str:
    """
    Save case text as a .txt file in data/cases/.
    Returns the saved file path.
    CommonLII returns HTML, not PDFs — we save as .txt
    which ingest.py can be extended to handle.
    """
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    filename = _safe_filename(title).replace(".pdf", ".txt")
    path     = os.path.join(DOWNLOAD_DIR, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"SOURCE: CommonLII\n")
        f.write(f"TITLE: {title}\n")
        f.write("=" * 60 + "\n\n")
        f.write(text)

    return path

# ── PUBLIC API ────────────────────────────────────────────────────────────────

def search(query: str, max_results: int = 10) -> list:
    """
    Search CommonLII for Singapore cases.

    Returns list of dicts:
    [{"title": "...", "url": "..."}, ...]
    """
    print(f"[COMMONLII] Searching: {query}")
    results = _search_commonlii(query, max_results=max_results)
    print(f"[COMMONLII] Found {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['title']}")
        print(f"     {r['url']}")
    return results

def download(results: list, max_download: int = 5) -> list:
    """
    Download and save case texts from search results.

    Args:
        results:      List from search()
        max_download: Maximum number of cases to download

    Returns:
        List of saved file paths
    """
    saved = []
    for i, result in enumerate(results[:max_download]):
        title = result["title"]
        url   = result["url"]
        print(f"\n[COMMONLII] Downloading {i+1}/{min(len(results), max_download)}: {title}")

        text = _fetch_case_text(url)
        if not text or len(text) < 200:
            print(f"  [SKIP] Insufficient text retrieved")
            continue

        path = _save_as_txt(title, text)
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
        print("[COMMONLII] No results found.")
        return []
    return download(results, max_download=max_download)


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "drug trafficking sentencing Singapore"
    paths = search_and_download(query, max_results=10, max_download=3)
    if paths:
        print(f"\n[DONE] {len(paths)} cases saved to data/cases/")
        print("Run python ingest.py to embed them into your database.")