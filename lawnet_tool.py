import os
import time
import re
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DOWNLOAD_DIR  = "data/cases"
REQUEST_DELAY = 2   # seconds between actions — be polite to the server
LAWNET_URL    = "https://www.lawnet.sg"
LOGIN_URL     = "https://www.lawnet.sg/lawnet/web/lawnet/home"
SEARCH_URL    = "https://www.lawnet.sg/lawnet/web/lawnet/free-resources"

# ── IMPORTANT NOTE ────────────────────────────────────────────────────────────
# This tool uses a visible (non-headless) browser so that YOU can:
# 1. Log in manually when prompted
# 2. Handle any CAPTCHAs or security checks
# 3. Re-authenticate when the session expires
# The agent pauses at each of these points and waits for your confirmation.
# ─────────────────────────────────────────────────────────────────────────────

def _wait_for_user(message: str):
    """Pause execution and wait for user to confirm before continuing."""
    print(f"\n[LAWNET] {message}")
    input("  Press Enter when ready to continue...")

def _safe_filename(name: str) -> str:
    """Convert a case name into a safe filename."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip().replace(" ", "_")
    return name[:100] + ".txt"

def _strip_html(html: str) -> str:
    """Strip HTML tags and clean whitespace."""
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'\s{3,}', '\n\n', text)
    return text.strip()

def _is_logged_out(page_text: str) -> bool:
    """Detect if Lawnet has timed out and logged the session out."""
    indicators = [
        "session has expired",
        "please log in",
        "login required",
        "sign in to continue",
        "your session"
    ]
    page_lower = page_text.lower()
    return any(ind in page_lower for ind in indicators)

# ── PLAYWRIGHT SESSION ────────────────────────────────────────────────────────

def _get_playwright():
    """Initialise Playwright with a visible browser window."""
    try:
        from playwright.sync_api import sync_playwright
        return sync_playwright
    except ImportError:
        print("[LAWNET] Playwright not installed.")
        print("  Run: pip install playwright")
        print("  Then: playwright install chromium")
        return None

def supervised_search(query: str, max_download: int = 5) -> list:
    """
    Open a visible browser, guide the user through a Lawnet research session,
    download relevant cases, and save them to data/cases/.

    This function PAUSES at key points and waits for the user to:
    - Log in to Lawnet
    - Re-authenticate if the session expires
    - Confirm before bulk downloading

    Args:
        query:        Research query to search on Lawnet
        max_download: Maximum number of cases to download

    Returns:
        List of saved file paths
    """
    sync_playwright = _get_playwright()
    if not sync_playwright is None:
        pass
    else:
        return []

    saved = []
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        # Launch VISIBLE browser — user must be able to see and interact
        browser = p.chromium.launch(headless=False, slow_mo=500)
        context = browser.new_context(accept_downloads=True)
        page    = context.new_page()

        try:
            # Step 1 — Navigate to Lawnet
            print(f"\n[LAWNET] Opening Lawnet in browser...")
            page.goto(LOGIN_URL, timeout=30000)
            time.sleep(2)

            # Step 2 — User logs in manually
            _wait_for_user(
                "Please log in to your Lawnet account in the browser window.\n"
                "  Complete the login fully before pressing Enter here."
            )

            # Step 3 — Verify login succeeded
            page_text = page.inner_text("body")
            if _is_logged_out(page_text):
                _wait_for_user(
                    "Login does not appear to have completed. "
                    "Please finish logging in, then press Enter."
                )

            # Step 4 — Navigate to search
            print(f"[LAWNET] Navigating to case search...")
            page.goto(f"{LAWNET_URL}/lawnet/web/lawnet/free-resources?p_p_id=legalresearchprovider", timeout=30000)
            time.sleep(2)

            # Step 5 — User performs the search
            _wait_for_user(
                f"Please search for: \"{query}\"\n"
                f"  Browse the results and open any cases you want downloaded.\n"
                f"  When you are ready to begin downloading, press Enter."
            )

            # Step 6 — Collect all open tabs as potential case pages
            print(f"[LAWNET] Checking open pages for case content...")
            all_pages = context.pages
            print(f"  Found {len(all_pages)} open page(s)")

            for i, case_page in enumerate(all_pages):
                if i >= max_download:
                    print(f"  [LIMIT] Reached maximum download limit of {max_download}")
                    break

                url       = case_page.url
                page_text = case_page.inner_text("body")

                # Check for session timeout on each page
                if _is_logged_out(page_text):
                    _wait_for_user(
                        "Session has expired on one of the pages. "
                        "Please log in again in the browser, then press Enter."
                    )
                    page_text = case_page.inner_text("body")

                # Skip non-case pages (search results, home page, etc.)
                if len(page_text) < 500:
                    continue
                if "lawnet.sg" not in url:
                    continue

                # Extract a title from the page
                try:
                    title = case_page.title().strip()
                    if not title or title == "LawNet":
                        title = f"lawnet_case_{i+1}"
                except Exception:
                    title = f"lawnet_case_{i+1}"

                # Save to file
                filename = _safe_filename(title)
                path     = os.path.join(DOWNLOAD_DIR, filename)

                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"SOURCE: Lawnet (supervised session)\n")
                    f.write(f"TITLE: {title}\n")
                    f.write(f"URL: {url}\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(page_text)

                print(f"  [SAVED] {filename} ({len(page_text)} chars)")
                saved.append(path)
                time.sleep(REQUEST_DELAY)

            # Step 7 — Confirm completion
            print(f"\n[LAWNET] Session complete. {len(saved)} case(s) saved.")
            _wait_for_user("Press Enter to close the browser.")

        except Exception as e:
            print(f"[LAWNET] Error during session: {e}")
            _wait_for_user("An error occurred. Press Enter to close the browser.")

        finally:
            browser.close()

    return saved


def install_playwright():
    """Helper to install Playwright if not already installed."""
    import subprocess
    print("[LAWNET] Installing Playwright...")
    subprocess.run(["pip", "install", "playwright"], check=True)
    subprocess.run(["playwright", "install", "chromium"], check=True)
    print("[LAWNET] Playwright installed successfully.")


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "drug trafficking sentencing"

    print(f"\n[LAWNET] Starting supervised research session")
    print(f"[LAWNET] Query: {query}")
    print(f"\nNote: This will open a real browser window.")
    print(f"You will need to log in to Lawnet manually.")
    confirm = input("Continue? (y/n): ").strip().lower()

    if confirm in ("y", "yes"):
        paths = supervised_search(query, max_download=5)
        if paths:
            print(f"\n[DONE] {len(paths)} case(s) saved to data/cases/")
            print("Run python ingest.py to embed them into your database.")
    else:
        print("Cancelled.")