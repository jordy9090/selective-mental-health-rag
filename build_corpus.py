
import json
import re
from pathlib import Path
from pypdf import PdfReader

RAW_DIR = Path("data/raw_docs")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = OUT_DIR / "chunks.jsonl"
DOC_INDEX_PATH = OUT_DIR / "doc_index.jsonl"


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
        return clean_text("\n".join(texts))
    except Exception as e:
        print(f"[WARN] failed to read pdf {path}: {e}")
        return ""


def read_txt(path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "cp949", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return clean_text(f.read())
        except Exception:
            continue
    print(f"[WARN] failed to read txt {path}")
    return ""


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    elif suffix == ".txt":
        return read_txt(path)
    else:
        print(f"[WARN] unsupported file type: {path}")
        return ""


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40):
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        if len(chunk.split()) >= 30:
            chunks.append(chunk)
        start += step

    return chunks


def main():
    doc_files = sorted(list(RAW_DIR.rglob("*.pdf")) + list(RAW_DIR.rglob("*.txt")))
    print(f"Found {len(doc_files)} documents (.pdf + .txt)")

    total_docs = 0
    total_chunks = 0

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f_chunks, \
         open(DOC_INDEX_PATH, "w", encoding="utf-8") as f_docs:

        for path in doc_files:
            source_family = path.parent.name   # coping / psychoeducation / safety
            title = path.stem
            text = read_document(path)

            if len(text.split()) < 80:
                print(f"[SKIP] too short: {path}")
                continue

            doc_id = f"{source_family}_{total_docs+1:03d}"
            chunks = chunk_text(text)

            if not chunks:
                print(f"[SKIP] no chunks: {path}")
                continue

            doc_record = {
                "doc_id": doc_id,
                "title": title,
                "source_family": source_family,
                "file_name": path.name,
                "file_path": str(path),
                "file_type": path.suffix.lower(),
                "num_chunks": len(chunks),
                "num_words": len(text.split()),
            }
            f_docs.write(json.dumps(doc_record, ensure_ascii=False) + "\n")

            for i, chunk in enumerate(chunks):
                chunk_record = {
                    "doc_id": doc_id,
                    "title": title,
                    "source_family": source_family,
                    "chunk_id": i,
                    "file_name": path.name,
                    "file_path": str(path),
                    "file_type": path.suffix.lower(),
                    "text": chunk,
                }
                f_chunks.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")

            total_docs += 1
            total_chunks += len(chunks)
            print(f"[OK] {doc_id} | {path.name} | chunks={len(chunks)}")

    print("\n=== DONE ===")
    print(f"documents processed: {total_docs}")
    print(f"chunks written: {total_chunks}")
    print(f"saved: {CHUNKS_PATH}")
    print(f"saved: {DOC_INDEX_PATH}")


if __name__ == "__main__":
    main()
