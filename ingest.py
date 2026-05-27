import os
import json
import hashlib
import requests
from pdfminer.high_level import extract_text

# ── CONFIG ──────────────────────────────────────────────────────────────────
SCAN_DIRS   = ["data/cases", "data/statutes", "data/notes"]
LOG_PATH    = "data/ingested_log.json"
FAISS_PATH  = "data/cases_db.json"
OLLAMA_URL  = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE  = 500
CHUNK_STEP  = 400

# ── HELPERS ─────────────────────────────────────────────────────────────────
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
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
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
    try:
        text = extract_text(path)
        if text and len(text.strip()) > 100:
            return text, "TEXT"
    except Exception:
        pass

    try:
        import pytesseract
        from pdf2image import convert_from_path
        pages = convert_from_path(path)
        ocr_text = ""
        for i, page in enumerate(pages):
            print(f"  [OCR] Page {i+1}/{len(pages)}...", flush=True)
            ocr_text += pytesseract.image_to_string(page) + "\n"
        return ocr_text, "OCR"
    except Exception as e:
        return "", f"FAIL ({e})"

# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    log = load_log()
    db  = load_db()

    processed = 0
    skipped   = 0
    failed    = 0

    for scan_dir in SCAN_DIRS:
        os.makedirs(scan_dir, exist_ok=True)
        print(f"\n[INGEST] Scanning: {scan_dir}\n", flush=True)

        pdf_files = [f for f in os.listdir(scan_dir) if f.lower().endswith(".pdf")]
        if not pdf_files:
            print(f"  (no PDF files found)\n", flush=True)
            continue

        for filename in sorted(pdf_files):
            path  = os.path.join(scan_dir, filename)
            fhash = file_hash(path)

            if fhash in log:
                print(f"File   : {filename}", flush=True)
                print(f"Status : SKIPPED\n", flush=True)
                skipped += 1
                continue

            text, method = extract_pdf_text(path)

            if not text.strip():
                print(f"File   : {filename}", flush=True)
                print(f"Mode   : {method}", flush=True)
                print(f"Status : FAILED — no text extracted\n", flush=True)
                failed += 1
                continue

            chunks     = chunk_text(text)
            total      = len(chunks)
            embed_ok   = 0
            embed_fail = 0

            print(f"File   : {filename}", flush=True)
            print(f"Mode   : {method}", flush=True)
            print(f"Chunks : {total}", flush=True)

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
                print(f"  Chunk {i+1}/{total} — {'OK' if vector else 'FAIL'}", flush=True)

            if embed_fail == total:
                print(f"Status : FAILED — all embeds failed (is Ollama running?)\n", flush=True)
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

            print(f"Status : COMPLETE\n", flush=True)
            processed += 1

    print(f"{'='*40}", flush=True)
    print(f"Processed : {processed}", flush=True)
    print(f"Skipped   : {skipped}", flush=True)
    print(f"Failed    : {failed}", flush=True)
    print(f"Vectors   : {len(db)} total in DB", flush=True)
    print(f"{'='*40}", flush=True)

if __name__ == "__main__":
    main()