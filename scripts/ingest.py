"""
ingest.py — Build the ChromaDB knowledge base from MOH CPG PDFs.
Ported 1:1 from FYP1 Colab notebook (MYDENTAL_CHATBOT_DATA_PRE_PROCESSING.ipynb).

Run ONCE to create ./chroma_db/ with embedded chunks:
    python scripts/ingest.py
"""

import os
import re
import glob
import shutil
from datetime import datetime

import pdfplumber
import tiktoken
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer


# ---------- Paths ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
COLLECTION_NAME = "cpg_guidelines"


# Step 1: Extract text + metadata 
def extract_text_with_metadata(pdf_path):
    filename = os.path.basename(pdf_path)
    title = filename.replace(".pdf", "").replace("_", " ")
    year = None
    full_text = ""

    # 1. Extract year from filename
    match = re.search(r"(19|20)\d{2}", filename)
    if match:
        year = match.group(0)

    # 2. If not found, search first page
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not year:
                try:
                    first_text = pdf.pages[0].extract_text()
                    if first_text:
                        m = re.search(r"(19|20)\d{2}", first_text)
                        if m:
                            year = m.group(0)
                except Exception:
                    pass

            # Fallback to current year
            if not year:
                year = str(datetime.now().year)

            # Extract full text page by page with [PAGE:X] markers
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    txt = page.extract_text()
                    if txt:
                        full_text += f"\n[PAGE:{page_num}]\n{txt}\n"
                except Exception:
                    full_text += f"\n[PAGE:{page_num}]\n[Text extraction failed]\n"

    except Exception as e:
        print(f"  ❌ Failed to open {filename}: {e}")
        if not year:
            year = str(datetime.now().year)

    # Category extraction (same logic as Colab)
    name_lower = title.lower()
    if "pregnant" in name_lower or "mother" in name_lower:
        category = "Prenatal Oral Health"
    elif "elderly" in name_lower:
        category = "Geriatric Dentistry"
    elif "preschool" in name_lower or "children" in name_lower:
        category = "Pediatric Dentistry"
    elif "cancer" in name_lower or "detection" in name_lower:
        category = "Oral Cancer"
    elif "cppc" in name_lower or "caries" in name_lower:
        category = "Caries Prevention"
    elif "periodontitis" in name_lower or "periodontal" in name_lower:
        category = "Periodontal Disease"
    elif "orthodontic" in name_lower:
        category = "Orthodontic"
    elif "fluoride" in name_lower:
        category = "Fluoride Programme"
    elif "brushing" in name_lower:
        category = "Tooth Brushing"
    else:
        category = "General Oral Health"

    metadata = {
        "title": title,
        "year": year,
        "category": category,
        "source_file": filename,
    }

    return full_text, metadata


# Step 2: Chunk by page 
def chunk_by_page(text, chunk_size_tokens=500, overlap_tokens=50):
    enc = tiktoken.get_encoding("cl100k_base")

    # Split by page
    page_pattern = r"\[PAGE:(\d+)\]\n(.*?)(?=\n\[PAGE:\d+\]|\Z)"
    pages = re.findall(page_pattern, text, re.DOTALL)

    all_chunks = []
    for page_num, page_text in pages:
        page_text = page_text.strip()
        if not page_text:
            continue

        # Tokenize the page text
        tokens = enc.encode(page_text)
        step = chunk_size_tokens - overlap_tokens
        if step <= 0:
            step = chunk_size_tokens

        start = 0
        while start < len(tokens):
            end = min(start + chunk_size_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            all_chunks.append(
                {
                    "text": chunk_text,
                    "page": page_num,
                    "token_length": len(chunk_tokens),
                }
            )
            start += step

    return all_chunks


# Main
def main():
    print("=" * 60)
    print("🔨 Building ChromaDB knowledge base")
    print("=" * 60)

    # 1. Find PDFs
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    pdf_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}")

    print(f"✅ Found {len(pdf_files)} PDFs:")
    for f in pdf_files:
        print(f"   - {os.path.basename(f)}")
    print()

    # 2. Extract text + metadata
    print("📄 Extracting text and metadata...")
    documents = []
    for pdf_file in pdf_files:
        text, meta = extract_text_with_metadata(pdf_file)
        documents.append({"text": text, "metadata": meta})
        print(
            f"   {meta['title']} — {meta['year']}, {meta['category']} — "
            f"{len(text):,} chars"
        )

    # 3. Chunk
    print("\n✂️  Chunking by page (500 tokens, 50 overlap)...")
    all_chunks = []
    all_metadatas = []

    for doc in documents:
        chunks_data = chunk_by_page(doc["text"])
        for idx, chunk_info in enumerate(chunks_data):
            all_chunks.append(chunk_info["text"])
            meta = doc["metadata"].copy()
            meta["chunk_index"] = idx
            meta["page_number"] = chunk_info["page"]
            meta["token_length"] = chunk_info["token_length"]
            all_metadatas.append(meta)

    print(f"   ✅ {len(all_chunks)} chunks total")

    # 4. Generate embeddings
    print(f"\n🧠 Generating embeddings with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    chunk_embeddings = model.encode(
        all_chunks, show_progress_bar=True, batch_size=32
    )
    print(
        f"   ✅ Generated {len(chunk_embeddings)} embeddings, "
        f"dim = {chunk_embeddings.shape[1]}"
    )

    # 5. Store in ChromaDB
    print(f"\n💾 Storing in ChromaDB at {CHROMA_PATH}...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH, ignore_errors=True)
        print("   Removed old DB folder")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.create_collection(
       name=COLLECTION_NAME,
       embedding_function=embed_fn,
       metadata={"hnsw:space": "cosine"},
   )

    batch_size = 100
    num_chunks = len(all_chunks)
    for i in range(0, num_chunks, batch_size):
        end = min(i + batch_size, num_chunks)
        collection.add(
            ids=[f"chunk_{j}" for j in range(i, end)],
            documents=all_chunks[i:end],
            embeddings=chunk_embeddings[i:end].tolist(),
            metadatas=all_metadatas[i:end],
        )
        print(f"   Added batch {i // batch_size + 1}/{(num_chunks - 1) // batch_size + 1}")

    print(
        f"\n✅ Stored {collection.count()} chunks at {CHROMA_PATH} "
        f"using {EMBEDDING_MODEL}"
    )
    print("=" * 60)
    print("🎉 Done! Next: python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()