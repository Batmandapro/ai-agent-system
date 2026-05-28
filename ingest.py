# FILE: ingest.py
# LOCATION: C:\Users\Admin\Desktop\ai-agent-system\ingest.py
# ACTION: Replace entire file

"""
Ingest pipeline for Singapore Legal AI system.
Scans configured directories for PDF and TXT files, extracts text using
MarkItDown (primary), pdfminer (secondary fallback), then Tesseract OCR
(final fallback for scanned image PDFs), chunks by paragraph boundaries,
embeds via Ollama nomic-embed-text, and saves results atomically to the
cases database.

Install requirements:
    pip install "markitdown[pdf]" pdfminer.six pytesseract pdf2image pillow

Tesseract (Windows):
    Download and install from https://github.com/UB-Mannheim/tesseract/wiki
    Default path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""

import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCAN_DIRS = [
    Path("data/cases"),
    Path("data/statutes"),
    Path("data/notes"),
]

LOG_PATH = Path("data/ingested_log.json")
DB_PATH  = Path("data/cases_db.json")

OLLAMA_URL  = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
MAX_WORKERS = 4

# Tesseract executable path (Windows default — adjust if installed elsewhere)
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Minimum characters for extracted text to be considered usable
MIN_TEXT_CHARS = 50

# Chunking parameters
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 1200

# ---------------------------------------------------------------------------
# Extractor availability flags
# ---------------------------------------------------------------------------

try:
    from markitdown import MarkItDown
    _MARKITDOWN = True
except ImportError:
    _MARKITDOWN = False
    print("WARNING: MarkItDown not installed — run: pip install \"markitdown[pdf]\"")

try:
    from pdfminer.high_level import extract_text as _pdfminer_extract
    _PDFMINER = True
except ImportError:
    _PDFMINER = False

try:
    import pytesseract
    from pdf2image import convert_from_path
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    _OCR = True
except ImportError:
    _OCR = False

# ---------------------------------------------------------------------------
# Text extraction — three-tier chain
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: Path) -> tuple[str, str]:
    """
    Extract text from a PDF using a three-tier fallback chain.

    Returns a tuple of (text, method) where method is one of:
        "markitdown"  — MarkItDown extracted usable text
        "pdfminer"    — pdfminer extracted usable text
        "ocr"         — Tesseract OCR extracted usable text
        "failed"      — all extractors returned empty or failed

    The OCR path is only reached when both MarkItDown and pdfminer
    return fewer than MIN_TEXT_CHARS characters — i.e. scanned image PDFs.
    """

    # ── Tier 1: MarkItDown ────────────────────────────────────────────────────
    if _MARKITDOWN:
        try:
            md   = MarkItDown()
            result = md.convert(str(file_path))
            text   = result.text_content or ""
            if len(text.strip()) >= MIN_TEXT_CHARS:
                print(f"  [TEXT] MarkItDown — clean text extracted successfully")
                return text, "markitdown"
            else:
                print(f"  [TEXT] MarkItDown returned too little text — trying pdfminer...")
        except Exception as exc:
            print(f"  [TEXT] MarkItDown failed: {exc} — trying pdfminer...")

    # ── Tier 2: pdfminer ─────────────────────────────────────────────────────
    if _PDFMINER:
        try:
            text = _pdfminer_extract(str(file_path)) or ""
            if len(text.strip()) >= MIN_TEXT_CHARS:
                print(f"  [TEXT] pdfminer — clean text extracted successfully")
                return text, "pdfminer"
            else:
                print(f"  [TEXT] pdfminer returned too little text — activating OCR...")
        except Exception as exc:
            print(f"  [TEXT] pdfminer failed: {exc} — activating OCR...")
    else:
        print(f"  [TEXT] pdfminer not available — activating OCR...")

    # ── Tier 3: Tesseract OCR ─────────────────────────────────────────────────
    if _OCR:
        print(f"  [OCR] Scanned PDF detected — this may take longer...")
        try:
            images = convert_from_path(str(file_path))
            text   = ""
            for i, img in enumerate(images):
                page_text = pytesseract.image_to_string(img)
                text += page_text
                print(f"  [OCR] Page {i + 1}/{len(images)} processed")
            if text.strip():
                print(f"  [OCR] OCR extraction successful")
                return text, "ocr"
            else:
                print(f"  [OCR] OCR returned empty text — file cannot be ingested")
                return "", "failed"
        except Exception as exc:
            print(f"  [OCR] OCR failed: {exc}")
            return "", "failed"
    else:
        print(
            f"  [OCR] pytesseract/pdf2image not installed — cannot OCR scanned PDF.\n"
            f"        Run: pip install pytesseract pdf2image pillow\n"
            f"        Then install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki"
        )
        return "", "failed"


def extract_text_from_file(file_path: Path) -> tuple[str, str]:
    """
    Dispatch text extraction based on file extension.
    Returns (text, method) — method is 'txt' for plain text files.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        try:
            return file_path.read_text(encoding="utf-8", errors="replace"), "txt"
        except Exception as exc:
            print(f"  Could not read {file_path.name}: {exc}")
            return "", "failed"
    elif suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    else:
        return "", "failed"


# ---------------------------------------------------------------------------
# Paragraph-boundary chunking
# ---------------------------------------------------------------------------

_PARA_NUM_RE      = re.compile(r"^\[(\d+)\]", re.MULTILINE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _detect_para_start(text: str) -> int | None:
    """Return the first Singapore legal paragraph number found, or None."""
    match = _PARA_NUM_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def _split_long_paragraph(paragraph: str) -> list[str]:
    """Split a paragraph exceeding MAX_CHUNK_CHARS at sentence boundaries."""
    sentences     = _SENTENCE_SPLIT_RE.split(paragraph)
    chunks: list  = []
    current_parts = []
    current_len   = 0

    for sentence in sentences:
        slen = len(sentence)
        if current_len + slen > MAX_CHUNK_CHARS and current_parts:
            chunk = " ".join(current_parts).strip()
            if len(chunk) >= MIN_CHUNK_CHARS:
                chunks.append(chunk)
            current_parts = []
            current_len   = 0
        current_parts.append(sentence)
        current_len += slen + 1

    if current_parts:
        remainder = " ".join(current_parts).strip()
        if remainder:
            if chunks and len(remainder) < MIN_CHUNK_CHARS:
                chunks[-1] = (chunks[-1] + " " + remainder).strip()
            else:
                chunks.append(remainder)

    return chunks if chunks else [paragraph]


def chunk_text(text: str) -> list[str]:
    """
    Split text into chunks on paragraph boundaries (blank lines).
    Merges short paragraphs and splits overly long ones.
    Paragraph markers such as [1], [2] are preserved verbatim.
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs     = [p.strip() for p in raw_paragraphs if p.strip()]

    merged = []
    buffer = ""

    for para in paragraphs:
        if not buffer:
            buffer = para
        elif len(buffer) < MIN_CHUNK_CHARS:
            buffer = buffer + "\n\n" + para
        else:
            merged.append(buffer)
            buffer = para

    if buffer:
        merged.append(buffer)

    final_chunks = []
    for chunk in merged:
        if len(chunk) > MAX_CHUNK_CHARS:
            final_chunks.extend(_split_long_paragraph(chunk))
        else:
            final_chunks.append(chunk)

    return final_chunks


# ---------------------------------------------------------------------------
# Embedding via Ollama
# ---------------------------------------------------------------------------

def embed_chunk(chunk_text_str: str) -> list[float]:
    """Embed a single text chunk via Ollama nomic-embed-text."""
    payload  = {"model": EMBED_MODEL, "prompt": chunk_text_str}
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["embedding"]


def embed_with_progress(chunks: list, file_name: str) -> list[tuple]:
    """
    Embed all chunks concurrently with a live progress bar.
    Returns list of (chunk_text, vector_or_None) tuples in original order.
    """
    total   = len(chunks)
    results = [None] * total
    done    = 0

    def _bar(d, t):
        filled = int(d / t * 30)
        bar    = "█" * filled + "░" * (30 - filled)
        sys.stdout.write(f"\r  Embedding [{bar}] {d}/{t} chunks")
        sys.stdout.flush()

    _bar(0, total)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(embed_chunk, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx   = future_to_idx[future]
            chunk = chunks[idx]
            try:
                results[idx] = (chunk, future.result())
            except Exception as exc:
                print(f"\n  Embedding failed for a chunk in {file_name}: {exc}")
                results[idx] = (chunk, None)
            done += 1
            _bar(done, total)

    sys.stdout.write("\n")
    sys.stdout.flush()
    return results


# ---------------------------------------------------------------------------
# Hash tracking and atomic saves
# ---------------------------------------------------------------------------

def md5_of_file(file_path: Path) -> str:
    hasher = hashlib.md5()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_json_file(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return default


def atomic_save_json(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_files() -> list[Path]:
    files = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            print(f"WARNING: Scan directory does not exist — skipping: {scan_dir}")
            continue
        for fp in sorted(scan_dir.rglob("*")):
            if fp.suffix.lower() in (".pdf", ".txt") and fp.is_file():
                files.append(fp)
    return files


def determine_folder_label(file_path: Path) -> str:
    for scan_dir in SCAN_DIRS:
        try:
            file_path.relative_to(scan_dir)
            return scan_dir.name
        except ValueError:
            continue
    return file_path.parent.name


# ---------------------------------------------------------------------------
# Single-file processing
# ---------------------------------------------------------------------------

def process_file(
    file_path: Path,
    folder_label: str,
    ingested_log: dict,
) -> tuple[list[dict], str | None]:
    """
    Extract, chunk, and embed all chunks from a single file.
    Returns (new_entries, file_hash).
    """
    file_hash = md5_of_file(file_path)

    if ingested_log.get(str(file_path)) == file_hash:
        print(f"  Skipping (unchanged): {file_path.name}")
        return [], file_hash

    print(f"\n{'─' * 50}")
    print(f"  Processing: {file_path.name}")

    raw_text, method = extract_text_from_file(file_path)

    if not raw_text.strip():
        print(f"  [FAIL] No readable text — skipping {file_path.name}")
        return [], file_hash

    chunks = chunk_text(raw_text)
    print(f"  Mode    : {method.upper()}")
    print(f"  Chunks  : {len(chunks)}")

    embedded     = embed_with_progress(chunks, file_path.name)
    new_entries  = []
    failed_embed = 0

    for chunk, vector in embedded:
        if vector is None:
            failed_embed += 1
            continue
        new_entries.append({
            "source":     str(file_path),
            "folder":     folder_label,
            "text":       chunk,
            "vector":     vector,
            "para_start": _detect_para_start(chunk),
        })

    if failed_embed:
        print(f"  {failed_embed} chunk(s) failed to embed.")

    print(f"  Saved   : {len(new_entries)} entries")
    return new_entries, file_hash


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("  Singapore Legal AI — Ingest Pipeline")
    print("=" * 55)

    extractors = []
    if _MARKITDOWN:
        extractors.append("MarkItDown (primary)")
    if _PDFMINER:
        extractors.append("pdfminer (fallback)")
    if _OCR:
        extractors.append("Tesseract OCR (scanned PDF fallback)")
    else:
        extractors.append("Tesseract OCR: NOT INSTALLED — scanned PDFs will be skipped")
    print(f"  Extractors : {' → '.join(extractors)}\n")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    ingested_log: dict = load_json_file(LOG_PATH, {})
    db_entries: list   = load_json_file(DB_PATH, [])

    files = collect_files()
    if not files:
        print("No PDF or TXT files found in any scan directory.")
        return

    print(f"  Found {len(files)} file(s) across scan directories.\n")

    total_new  = 0
    processed  = 0
    skipped    = 0
    failed     = 0

    for file_path in files:
        folder_label = determine_folder_label(file_path)
        new_entries, file_hash = process_file(
            file_path, folder_label, ingested_log
        )

        if new_entries:
            db_entries.extend(new_entries)
            ingested_log[str(file_path)] = file_hash
            total_new += len(new_entries)
            processed += 1
            atomic_save_json(DB_PATH, db_entries)
            atomic_save_json(LOG_PATH, ingested_log)

        elif ingested_log.get(str(file_path)) == file_hash:
            skipped += 1

        else:
            # No text extracted — record hash so we do not retry on every run
            failed += 1
            if file_hash:
                ingested_log[str(file_path)] = file_hash
                atomic_save_json(LOG_PATH, ingested_log)

    print(f"\n{'=' * 55}")
    print(f"  Ingest complete")
    print(f"  Processed  : {processed} file(s)")
    print(f"  Skipped    : {skipped} (unchanged)")
    print(f"  Failed     : {failed} (no text extracted)")
    print(f"  New entries: {total_new}")
    print(f"  DB total   : {len(db_entries)} entries")
    print(f"  DB path    : {DB_PATH}")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
