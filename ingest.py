import os
from pdfminer.high_level import extract_text
import pytesseract
from pdf2image import convert_from_path

from legal_faiss import LegalFAISS

rag = LegalFAISS()


def extract_pdf_text(path):
    try:
        text = extract_text(path)
        if text and len(text.strip()) > 50:
            return text
    except:
        pass

    # OCR fallback
    images = convert_from_path(path)
    text = ""

    for img in images:
        text += pytesseract.image_to_string(img)

    return text


def chunk_text(text, size=800):
    words = text.split()
    return [
        " ".join(words[i:i+size])
        for i in range(0, len(words), size)
    ]


def ingest_folder(folder, source_type="case"):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        print(f"\n[FILE] {file}")

        text = extract_pdf_text(path)
        chunks = chunk_text(text)

        print(f"[CHUNKS] {len(chunks)}")

        for i, chunk in enumerate(chunks):
            rag.add(
                chunk,
                source_type,
                {
                    "source": file,
                    "chunk": i
                }
            )

    print("\n[INGESTION COMPLETE]")


if __name__ == "__main__":
    ingest_folder("data/cases")