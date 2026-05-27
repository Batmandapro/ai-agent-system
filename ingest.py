import os
import json
import hashlib
import requests
import sys
from pdfminer.high_level import extract_text

# ── CONFIG ────────────────────────────────────────────────────────────────────
SCAN_DIRS   = ["data/cases", "data/statutes", "data/notes"]
LOG_PATH    = "data/ingested_log.json"
FAISS_PATH  = "data/cases_db.json"
OLLAMA_URL  = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE  = 500   # characters per chunk
CHUNK_STEP  = 400   # step size — overlap = CHUNK_SIZE minus CHUNK_STEP

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
    with open(FAISS_PATH, "w") as f:
        json.dump(db, f, indent=2)

def chunk_text(text):
    chunks = []
    start  = 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_STEP
    return chunks

def embed(text):
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

def extract_pdf_text(path):
    """Extract text from PDF using pdfminer; fall back to OCR if needed."""
    try:
        text = extract_text(path)
        if text and len(text.strip()) > 100:
            return text, "TEXT"
    except Exception:
        pass

    # OCR fallback for scanned PDFs
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
    """Extract text from plain .txt files (e.g. downloaded from CommonLII or eLitigation)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), "TEXT"
    except Exception as e:
        return "", f"FAIL ({e})"

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log = load_log()
    db  = load_db()

    processed = 0
    skipped   = 0
    failed    = 0

    for scan_dir in SCAN_DIRS:
        os.makedirs(scan_dir, exist_ok=True)
        print(f"\n[INGEST] Scanning: {scan_dir}\n")

        # Accept both PDF and TXT files
        all_files = [
            f for f in os.listdir(scan_dir)
            if f.lower().endswith(".pdf") or f.lower().endswith(".txt")
        ]

        if not all_files:
            print("  (no files found)\n")
            continue

        for filename in sorted(all_files):
            path  = os.path.join(scan_dir, filename)
            fhash = file_hash(path)

            if fhash in log:
                print(f"File   : {filename}")
                print(f"Status : SKIPPED\n")
                skipped += 1
                continue

            # Extract text based on file type
            if filename.lower().endswith(".txt"):
                text, method = extract_txt_text(path)
            else:
                text, method = extract_pdf_text(path)

            if not text.strip():
                print(f"File   : {filename}")
                print(f"Mode   : {method}")
                print(f"Status : FAILED — no text extracted\n")
                failed += 1
                continue

            chunks     = chunk_text(text)
            total      = len(chunks)
            embed_ok   = 0
            embed_fail = 0

            print(f"File   : {filename}")
            print(f"Mode   : {method}")
            print(f"Chunks : {total}")

            for i, chunk in enumerate(chunks):
                vector = embed(chunk)
                if vector:
                    db.append({
                        "source": filename,
                        "folder": scan_dir,
                        "chunk":  chunk,
                        "vector": vector
                    })
                    embed_ok += 1
                else:
                    embed_fail += 1

                # Single-line progress bar — overwrites itself
                bar_done  = int((i + 1) / total * 30)
                bar_empty = 30 - bar_done
                bar       = "█" * bar_done + "░" * bar_empty
                sys.stdout.write(f"\r  [{bar}] {i+1}/{total} chunks")
                sys.stdout.flush()

            # Move past the progress bar line
            sys.stdout.write("\n")
            sys.stdout.flush()

            if embed_fail == total:
                print(f"Status : FAILED — all embeds failed (is Ollama running?)\n")
                failed += 1
                continue

            log[fhash] = {
                "file":       filename,
                "folder":     scan_dir,
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