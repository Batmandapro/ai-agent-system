import os
import json
import hashlib
import requests
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pdfminer.high_level import extract_text

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCAN_DIRS   = [
    "data/cases",
    "data/statutes",
    "data/notes",
]

LOG_PATH    = "data/ingested_log.json"
FAISS_PATH  = "data/cases_db.json"
OLLAMA_URL  = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

CHUNK_SIZE  = 1000
CHUNK_STEP  = 800
MAX_WORKERS = 4

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    return {}

def save_log(log):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)

def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def load_db():
    if os.path.exists(FAISS_PATH):
        with open(FAISS_PATH, "r") as f:
            return json.load(f)
    return []

def save_db(db):
    os.makedirs(os.path.dirname(FAISS_PATH), exist_ok=True)
    tmp = FAISS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(db, f, indent=2)
    os.replace(tmp, FAISS_PATH)

def chunk_text(text):
    """Split text into overlapping chunks of CHUNK_SIZE with CHUNK_STEP stride."""
    chunks = []
    start  = 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_STEP
    return chunks

def embed_single(text):
    """Embed a single chunk via Ollama. Returns vector or None on failure."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json().get("embedding")
    except Exception:
        return None

def embed_concurrent(chunks, max_workers=MAX_WORKERS):
    """Embed a list of chunks using a thread pool for concurrency."""
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
            sys.stdout.write(f"\r  [{bar}] {completed}/{total} chunks")
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return results

def extract_pdf_text(path):
    """Extract text from PDF using pdfminer; fall back to OCR if needed."""
    try:
        text = extract_text(path)
        if text and len(text.strip()) > 100:
            return text, "TEXT"
    except Exception:
        pass
    try:
        import pytesseract
        from pdf2image import convert_from_path
        pages    = convert_from_path(path)
        ocr_text = ""
        for i, page in enumerate(pages):
            sys.stdout.write(f"\r  [OCR] Page {i+1}/{len(pages)}...")
            sys.stdout.flush()
            ocr_text += pytesseract.image_to_string(page) + "\n"
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()
        return ocr_text, "OCR"
    except Exception as e:
        return "", f"FAIL ({e})"

def extract_txt_text(path):
    """Extract text from plain .txt files."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), "TEXT"
    except Exception as e:
        return "", f"FAIL ({e})"

def collect_files(scan_dirs):
    """Recursively collect all PDF and TXT files from each directory."""
    collected = []
    for scan_dir in scan_dirs:
        os.makedirs(scan_dir, exist_ok=True)
        for root, dirs, files in os.walk(scan_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for filename in sorted(files):
                if filename.lower().endswith(".pdf") or filename.lower().endswith(".txt"):
                    full_path    = os.path.join(root, filename)
                    display_path = os.path.relpath(full_path)
                    collected.append((full_path, display_path, filename))
    return collected

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log = load_log()
    db  = load_db()

    processed = 0
    skipped   = 0
    failed    = 0

    print(f"\n[INGEST] Scanning directories: {SCAN_DIRS}\n")
    all_files = collect_files(SCAN_DIRS)

    if not all_files:
        print("  (no files found in any scan directory)\n")
    else:
        print(f"  Found {len(all_files)} file(s) total\n")

    for full_path, display_path, filename in all_files:
        fhash = file_hash(full_path)

        if fhash in log:
            print(f"File   : {display_path}")
            print(f"Status : SKIPPED (already ingested)\n")
            skipped += 1
            continue

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
                    "text":   chunk,      # NOTE: key is "text" not "chunk"
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
        save_log(log)
        save_db(db)

        status = "COMPLETE" if embed_fail == 0 else f"COMPLETE ({embed_fail} embed errors)"
        print(f"Status : {status}\n")
        processed += 1

    print("=" * 40)
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(f"Vectors   : {len(db)} total in DB")
    print("=" * 40)


if __name__ == "__main__":
    main()