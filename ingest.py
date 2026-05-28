# FILE: ingest.py
# REPLACES: C:\Users\Admin\Desktop\ai-agent-system\ingest.py

"""
Ingest pipeline for Singapore Legal AI system.
Scans configured directories for PDF and TXT files, extracts text using
MarkItDown (primary) or pdfminer (fallback), chunks by paragraph boundaries,
embeds via Ollama, and saves results atomically to the cases database.

Install requirement:
    pip install "markitdown[pdf]"
"""

import hashlib
import json
import os
import re
import sys
import time
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

# Chunking parameters
MIN_CHUNK_CHARS = 200
MAX_CHUNK_CHARS = 1200

# ---------------------------------------------------------------------------
# MarkItDown import with graceful fallback to pdfminer
# ---------------------------------------------------------------------------

try:
    from markitdown import MarkItDown
    _MARKITDOWN_AVAILABLE = True
except ImportError:
    _MARKITDOWN_AVAILABLE = False
    print(
        "WARNING: MarkItDown is not installed. "
        "Falling back to pdfminer for PDF extraction. "
        "Run:  pip install \"markitdown[pdf]\"  to enable the primary extractor."
    )

if not _MARKITDOWN_AVAILABLE:
    try:
        from pdfminer.high_level import extract_text as _pdfminer_extract
        _PDFMINER_AVAILABLE = True
    except ImportError:
        _PDFMINER_AVAILABLE = False
        print(
            "WARNING: pdfminer is also not installed. "
            "PDF files will be skipped entirely."
        )

# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file.

    Primary:  MarkItDown — returns clean Markdown preserving headings,
              numbered lists and paragraph structure. The raw Markdown is
              returned as-is so that downstream citation extraction can
              identify paragraph markers such as [1], [2], [45].

    Fallback: pdfminer.high_level.extract_text if MarkItDown is unavailable.

    Returns an empty string if neither extractor is available or if
    extraction fails.
    """
    if _MARKITDOWN_AVAILABLE:
        try:
            md = MarkItDown()
            result = md.convert(str(file_path))
            # Return the full Markdown text without stripping any markers
            return result.text_content
        except Exception as exc:
            print(f"  MarkItDown failed for {file_path.name}: {exc}. "
                  "Attempting pdfminer fallback.")

    # Fallback path
    if _PDFMINER_AVAILABLE:
        try:
            return _pdfminer_extract(str(file_path)) or ""
        except Exception as exc:
            print(f"  pdfminer failed for {file_path.name}: {exc}.")
            return ""

    print(f"  No PDF extractor available — skipping {file_path.name}.")
    return ""


def extract_text_from_file(file_path: Path) -> str:
    """
    Dispatch text extraction based on file extension.
    TXT files are read directly; PDFs use the extractor chain above.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            print(f"  Could not read {file_path.name}: {exc}")
            return ""
    elif suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    else:
        return ""

# ---------------------------------------------------------------------------
# Paragraph-boundary chunking
# ---------------------------------------------------------------------------

# Matches Singapore legal paragraph markers: [1], [12], [123] at line start.
_PARA_NUM_RE = re.compile(r"^\[(\d+)\]", re.MULTILINE)

# Sentence boundary: period (or ! or ?) followed by one or more spaces then a
# capital letter — used when a single paragraph exceeds MAX_CHUNK_CHARS.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _detect_para_start(text: str) -> int | None:
    """
    Return the first Singapore legal paragraph number found in *text*,
    or None if no such marker is present.
    """
    match = _PARA_NUM_RE.search(text)
    if match:
        return int(match.group(1))
    return None


def _split_long_paragraph(paragraph: str) -> list[str]:
    """
    Split a paragraph that exceeds MAX_CHUNK_CHARS at sentence boundaries.
    Sentences are accumulated until adding the next would exceed the limit;
    at that point the current accumulation is emitted as a chunk.
    Any remainder shorter than MIN_CHUNK_CHARS is merged into the last chunk.
    """
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len + sentence_len > MAX_CHUNK_CHARS and current_parts:
            chunk = " ".join(current_parts).strip()
            if len(chunk) >= MIN_CHUNK_CHARS:
                chunks.append(chunk)
                current_parts = []
                current_len = 0
        current_parts.append(sentence)
        current_len += sentence_len + 1  # +1 for the joining space

    # Emit whatever remains.
    if current_parts:
        remainder = " ".join(current_parts).strip()
        if remainder:
            if chunks and len(remainder) < MIN_CHUNK_CHARS:
                # Merge short remainder into the last chunk.
                chunks[-1] = (chunks[-1] + " " + remainder).strip()
            else:
                chunks.append(remainder)

    return chunks if chunks else [paragraph]


def chunk_text(text: str) -> list[str]:
    """
    Split text into chunks on paragraph boundaries (blank lines in Markdown).

    Rules:
    - A paragraph boundary is one or more consecutive blank lines.
    - Paragraphs shorter than MIN_CHUNK_CHARS are merged with the next
      paragraph until the combined length meets the minimum or all
      paragraphs are exhausted.
    - Paragraphs longer than MAX_CHUNK_CHARS are split at sentence
      boundaries (see _split_long_paragraph).
    - The Markdown structure (# headings, [n] markers, numbered lists) is
      preserved verbatim within each chunk.
    """
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    merged: list[str] = []
    buffer = ""

    for para in paragraphs:
        if not buffer:
            buffer = para
        else:
            combined = buffer + "\n\n" + para
            if len(buffer) < MIN_CHUNK_CHARS:
                buffer = combined
            else:
                merged.append(buffer)
                buffer = para

    if buffer:
        merged.append(buffer)

    final_chunks: list[str] = []
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
    """
    Send a text chunk to the local Ollama embedding endpoint and return
    the embedding vector as a list of floats.
    Raises requests.RequestException on network or server errors.
    """
    payload = {"model": EMBED_MODEL, "prompt": chunk_text_str}
    response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["embedding"]


def embed_with_progress(chunks: list[str], file_name: str) -> list[tuple[str, list[float] | None]]:
    """
    Embed a list of chunks concurrently using a thread pool, displaying a
    live progress bar in the terminal.

    Returns a list of (chunk_text, vector_or_None) tuples in submission order.
    """
    total = len(chunks)
    results: list[tuple[str, list[float] | None]] = [None] * total  # type: ignore
    completed = 0

    def _render_bar(done: int, total: int) -> None:
        bar_done  = int(done / total * 30)
        bar_empty = 30 - bar_done
        bar       = "█" * bar_done + "░" * bar_empty
        sys.stdout.write(f"\r  Embedding [{bar}] {done}/{total} chunks")
        sys.stdout.flush()

    _render_bar(0, total)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(embed_chunk, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            chunk = chunks[idx]
            try:
                vector = future.result()
                results[idx] = (chunk, vector)
            except Exception as exc:
                print(f"\n  Embedding failed for a chunk in {file_name}: {exc}")
                results[idx] = (chunk, None)
            completed += 1
            _render_bar(completed, total)

    sys.stdout.write("\n")
    sys.stdout.flush()
    return results

# ---------------------------------------------------------------------------
# Hash tracking and atomic saves
# ---------------------------------------------------------------------------

def md5_of_file(file_path: Path) -> str:
    """Compute the MD5 digest of a file's contents in 64 KB blocks."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_json_file(path: Path, default):
    """Load a JSON file, returning *default* if the file does not exist."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return default


def atomic_save_json(path: Path, data) -> None:
    """
    Serialise *data* to JSON and save it atomically by writing to a
    temporary file then replacing the target path.
    """
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------

def collect_files() -> list[Path]:
    """
    Walk each directory in SCAN_DIRS and collect all PDF and TXT files.
    Directories that do not exist are skipped with a warning.
    """
    files: list[Path] = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            print(f"WARNING: Scan directory does not exist — skipping: {scan_dir}")
            continue
        for file_path in sorted(scan_dir.rglob("*")):
            if file_path.suffix.lower() in (".pdf", ".txt") and file_path.is_file():
                files.append(file_path)
    return files


def determine_folder_label(file_path: Path) -> str:
    """
    Return a short label identifying which top-level scan directory
    the file belongs to (e.g. 'cases', 'statutes', 'notes').
    Falls back to the immediate parent directory name.
    """
    for scan_dir in SCAN_DIRS:
        try:
            file_path.relative_to(scan_dir)
            return scan_dir.name
        except ValueError:
            continue
    return file_path.parent.name


def process_file(
    file_path: Path,
    folder_label: str,
    ingested_log: dict,
    db_entries: list,
) -> tuple[list[dict], str | None]:
    """
    Extract, chunk, and embed all chunks from a single file.

    Returns:
        (new_entries, file_hash)  — new_entries is a (possibly empty) list of
        DB records to append; file_hash is the MD5 of the file (or None on
        extraction failure).
    """
    file_hash = md5_of_file(file_path)

    if ingested_log.get(str(file_path)) == file_hash:
        print(f"  Skipping (unchanged): {file_path.name}")
        return [], file_hash

    print(f"\n  Processing: {file_path.name}")
    raw_text = extract_text_from_file(file_path)

    if not raw_text.strip():
        print(f"  No text extracted from {file_path.name} — skipping.")
        return [], file_hash

    chunks = chunk_text(raw_text)
    print(f"  Chunks: {len(chunks)}")

    embedded = embed_with_progress(chunks, file_path.name)

    new_entries: list[dict] = []
    failed = 0

    for chunk, vector in embedded:
        if vector is None:
            failed += 1
            continue
        para_start = _detect_para_start(chunk)
        entry = {
            "source":     str(file_path),
            "folder":     folder_label,
            "text":       chunk,          # Key must remain "text" — do not rename
            "vector":     vector,
            "para_start": para_start,
        }
        new_entries.append(entry)

    if failed:
        print(f"  {failed} chunk(s) failed to embed in {file_path.name}.")

    return new_entries, file_hash


def main() -> None:
    print("=== Singapore Legal AI — Ingest Pipeline ===")
    if _MARKITDOWN_AVAILABLE:
        print("PDF extractor : MarkItDown (primary)")
    elif _PDFMINER_AVAILABLE:
        print("PDF extractor : pdfminer (fallback)")
    else:
        print("PDF extractor : NONE — PDF files will be skipped")

    # Ensure output directories exist.
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing state.
    ingested_log: dict = load_json_file(LOG_PATH, {})
    db_entries: list   = load_json_file(DB_PATH, [])

    files = collect_files()
    if not files:
        print("No PDF or TXT files found in any scan directory.")
        return

    print(f"\nFound {len(files)} file(s) across scan directories.\n")

    total_new_entries = 0
    files_processed   = 0
    files_skipped     = 0

    for file_path in files:
        folder_label = determine_folder_label(file_path)
        new_entries, file_hash = process_file(
            file_path, folder_label, ingested_log, db_entries
        )

        if new_entries:
            db_entries.extend(new_entries)
            ingested_log[str(file_path)] = file_hash
            total_new_entries += len(new_entries)
            files_processed   += 1

            # Save incrementally after each file to minimise data loss if the
            # process is interrupted.
            atomic_save_json(DB_PATH, db_entries)
            atomic_save_json(LOG_PATH, ingested_log)
            print(f"  Saved {len(new_entries)} new entry/entries to database.")

        elif file_hash and ingested_log.get(str(file_path)) == file_hash:
            files_skipped += 1

        else:
            # Extraction produced no text — record the hash so the file is not
            # re-attempted on every run unless it changes on disk.
            if file_hash:
                ingested_log[str(file_path)] = file_hash
                atomic_save_json(LOG_PATH, ingested_log)

    print("\n=== Ingest complete ===")
    print(f"  Files processed : {files_processed}")
    print(f"  Files skipped   : {files_skipped} (unchanged)")
    print(f"  New DB entries  : {total_new_entries}")
    print(f"  DB total entries: {len(db_entries)}")
    print(f"  DB path         : {DB_PATH}")


if __name__ == "__main__":
    main()