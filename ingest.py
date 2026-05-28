import os
import json
import hashlib
import requests
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdfminer.high_level import extract_text
from typing import List, Dict, Any, Tuple, Union

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCAN_DIRS: List[str] = [
    "data/cases",
    "data/statutes",
    "data/notes",
]

LOG_PATH    = "data/ingested_log.json"
DB_PATH     = "data/cases_db.json"  # Renamed from FAISS_PATH for clarity
OLLAMA_URL  = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE  = 1000
CHUNK_STEP  = 800
MAX_WORKERS = 4

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_log() -> Dict[str, Any]:
    """Loads the ingestion log from the specified path."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_log(log: Dict[str, Any]):
    """Saves the ingestion log atomically to the specified path."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    tmp_path = LOG_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    os.replace(tmp_path, LOG_PATH)

def file_hash(path: str) -> str:
    """Generates an MD5 hash of a file using buffered reading."""
    h = hashlib.md5()
    # Read file in chunks to handle large files efficiently
    with open(path, "rb") as f:
        while chunk := f.read(8192):  # Read in 8KB chunks
            h.update(chunk)
    return h.hexdigest()

def load_db() -> List[Dict[str, Any]]:
    """Loads the custom JSON vector database from the specified path."""
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_db(db: List[Dict[str, Any]]):
    """Saves the custom JSON vector database atomically to the specified path."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    tmp_path = DB_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)
    os.replace(tmp_path, DB_PATH)

def chunk_text(text: str) -> List[str]:
    """Splits text into overlapping chunks of CHUNK_SIZE with CHUNK_STEP stride."""
    chunks = []
    start  = 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_STEP
    return chunks

def embed_single(text: str) -> Union[List[float], None]:
    """Embeds a single text chunk via Ollama. Returns vector or None on failure."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json().get("embedding")
    except Exception:
        # Log the exception for debugging if needed, but for now, just return None
        return None

def embed_concurrent(chunks: List[str], max_workers: int = MAX_WORKERS) -> List[Union[List[float], None]]:
    """Embeds a list of chunks using a thread pool for concurrency, with a progress bar."""
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(embed_single, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        completed = 0
        total     = len(chunks)
        for future in as_completed(future_to_idx):
            idx            = future_to_idx[future]
            results[idx]   = future.result()
            completed     += 1
            bar_done  = int(completed / total * 30)
            bar_empty = 30 - bar_done
            bar       = "█" * bar_done + "░" * bar_empty
            sys.stdout.write(f"\r  [{bar}] {completed}/{total} chunks embedded") # Updated message
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return results

def extract_pdf_text(path: str) -> Tuple[str, str]:
    """Extracts text from PDF using pdfminer; falls back to OCR if needed."""
    try:
        text = extract_text(path)
        if text and len(text.strip()) > 100: # Heuristic for sufficient text
            return text, "TEXT"
    except Exception:
        pass # Fall through to OCR if pdfminer fails or finds too little text
    try:
        import pytesseract
        from pdf2image import convert_from_path
        pages    = convert_from_path(path)
        ocr_text = ""
        for i, page in enumerate(pages):
            sys.stdout.write(f"\r  [OCR] Processing page {i+1} of {len(pages)}...") # Updated message
            sys.stdout.flush()
            ocr_text += pytesseract.image_to_string(page) + "\n"
        sys.stdout.write("\r" + " " * 40 + "\r") # Clear the OCR progress line
        sys.stdout.flush()
        return ocr_text, "OCR"
    except Exception as e:
        return "", f"FAIL ({e})"

def extract_txt_text(path: str) -> Tuple[str, str]:
    """Extracts text from plain .txt files."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), "TEXT"
    except Exception as e:
        return "", f"FAIL ({e})"

def collect_files(scan_dirs: List[str]) -> List[Tuple[str, str, str]]:
    """Recursively collects all PDF and TXT files from each specified directory."""
    collected = []
    for scan_dir in scan_dirs:
        os.makedirs(scan_dir, exist_ok=True)
        for root, dirs, files in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")] # Exclude hidden directories
            for filename in sorted(files):
                if filename.lower().endswith(".pdf") or filename.lower().endswith(".txt"):
                    full_path    = os.path.join(root, filename)
                    display_path = os.path.relpath(full_path)
                    collected.append((full_path, display_path, filename))
    return collected

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    """Main ingestion function to process files, embed text, and update the database."""
    log = load_log()
    db  = load_db()

    processed = 0
    skipped   = 0
    failed    = 0

    print(f"\n[INGEST] Scanning directories: {SCAN_DIRS}\n")
    all_files = collect_files(SCAN_DIRS)

    if not all_files:
        print("  (No files found in any scan directory)\n") # Updated phrasing
    else:
        print(f"  Found {len(all_files)} file(s) in total\n") # Updated phrasing

    for full_path, display_path, filename in all_files:
        fhash = file_hash(full_path)

        if fhash in log:
            print(f"File   : {display_path}")
            print(f"Status : SKIPPED (already ingested)\n")
            skipped += 1
            continue

        text: str
        method: str
        if filename.lower().endswith(".txt"):
            text, method = extract_txt_text(full_path)
        else:
            text, method = extract_pdf_text(full_path)

        if not text.strip():
            print(f"File   : {display_path}")
            print(f"Mode   : {method}")
            print(f"Status : FAILED — no text extracted\n")
            failed += 1
            continue

        chunks = chunk_text(text)
        total  = len(chunks)

        print(f"File   : {display_path}")
        print(f"Mode   : {method}")
        print(f"Chunks : {total}")

        vectors    = embed_concurrent(chunks, max_workers=MAX_WORKERS)
        embed_ok   = 0
        embed_fail = 0

        for chunk, vector in zip(chunks, vectors):
            if vector:
                db.append({
                    "source": filename,
                    "folder": os.path.dirname(full_path),
                    "text":   chunk,
                    "vector": vector
                })
                embed_ok += 1
            else:
                embed_fail += 1

        if embed_fail == total:
            print(f"Status : FAILED — all embeds failed (is Ollama running?)\n")
            failed += 1
            continue

        log[fhash] = {
            "file":       filename,
            "folder":     os.path.dirname(full_path),
            "method":     method,
            "chunks":     total,
            "embed_ok":   embed_ok,
            "embed_fail": embed_fail
        }
        # Save log and database after each file for robustness against crashes.
        save_log(log)
        save_db(db)

        status = "COMPLETE" if embed_fail == 0 else f"COMPLETE ({embed_fail} embed errors)"
        print(f"Status : {status}\n")
        processed += 1

    print("=" * 40)
    print(f"Processed files : {processed}") # Updated phrasing
    print(f"Skipped files   : {skipped}")   # Updated phrasing
    print(f"Failed files    : {failed}")    # Updated phrasing
    print(f"Total vectors   : {len(db)} in database") # Updated phrasing
    print("=" * 40)


if __name__ == "__main__":
    main()