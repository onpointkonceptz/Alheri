"""
app/rag.py
Knowledge Base Ingestion and Local Vector Retrieval Engine for Alheri.
Operates 100% offline with zero cloud telemetry.
Uses Ollama (nomic-embed-text) and ChromaDB persistent storage.
"""

import os
import glob
import re
import time
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import ollama

# Default configuration paths
DEFAULT_KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "knowledge_base"))
DEFAULT_STORE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_store"))
EMBEDDING_MODEL = "nomic-embed-text"
COLLECTION_NAME = "alheri_kb"


def get_chroma_client(store_dir: str = DEFAULT_STORE_DIR) -> chromadb.ClientAPI:
    """Returns a persistent ChromaDB client pointing to store_dir with telemetry disabled."""
    os.makedirs(store_dir, exist_ok=True)
    # Ensure telemetry is completely off
    settings = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    )
    return chromadb.PersistentClient(path=store_dir, settings=settings)


def get_kb_collection(store_dir: str = DEFAULT_STORE_DIR):
    """Returns or creates the ChromaDB collection configured for cosine similarity."""
    client = get_chroma_client(store_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def extract_chunks_from_markdown(file_path: str) -> List[Dict[str, Any]]:
    """
    Parses a Markdown file into semantically meaningful chunks based on headers.
    Maintains document hierarchy and header metadata for accurate triage citation.
    """
    filename = os.path.basename(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by major headings (## or ###)
    lines = content.splitlines()
    doc_title = filename
    for line in lines:
        if line.startswith("# "):
            doc_title = line.replace("# ", "").strip()
            break

    chunks = []
    current_section = doc_title
    current_lines: List[str] = []

    def commit_chunk(lines_list: List[str], sec: str):
        text = "\n".join(lines_list).strip()
        if len(text) > 80:  # Skip trivial fragments
            chunks.append({
                "source": filename,
                "document_title": doc_title,
                "section": sec,
                "text": text,
            })

    for line in lines:
        if re.match(r"^#{1,3}\s+", line):
            if current_lines:
                commit_chunk(current_lines, current_section)
                current_lines = []
            current_section = re.sub(r"^#{1,3}\s+", "", line).strip()
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        commit_chunk(current_lines, current_section)

    return chunks


def compute_embedding(text: str, model: str = EMBEDDING_MODEL) -> List[float]:
    """Generates an embedding vector using the local Ollama nomic-embed-text model."""
    resp = ollama.embed(model=model, input=text)
    if hasattr(resp, "embeddings"):
        return resp.embeddings[0]
    elif isinstance(resp, dict) and "embeddings" in resp:
        return resp["embeddings"][0]
    raise RuntimeError(f"Unexpected response structure from ollama.embed: {resp}")


def ingest_knowledge_base(
    kb_dir: str = DEFAULT_KB_DIR,
    store_dir: str = DEFAULT_STORE_DIR
) -> Dict[str, Any]:
    """
    Indexes all Markdown files in kb_dir into ChromaDB vector store.
    Uses local nomic-embed-text model via Ollama.
    """
    start_time = time.time()
    md_files = glob.glob(os.path.join(kb_dir, "*.md"))
    if not md_files:
        return {
            "status": "warning",
            "message": f"No markdown files found in {kb_dir}",
            "files_indexed": 0,
            "chunks_indexed": 0,
            "duration_seconds": round(time.time() - start_time, 3)
        }

    client = get_chroma_client(store_dir)
    # Recreate collection to ensure clean state upon manual re-ingest
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks = []
    for fpath in md_files:
        chunks = extract_chunks_from_markdown(fpath)
        all_chunks.extend(chunks)

    if not all_chunks:
        return {
            "status": "warning",
            "message": "Files found but no chunks extracted.",
            "files_indexed": len(md_files),
            "chunks_indexed": 0,
            "duration_seconds": round(time.time() - start_time, 3)
        }

    # Embed each chunk
    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for idx, ch in enumerate(all_chunks):
        chunk_id = f"{ch['source']}_{idx:03d}"
        formatted_content = f"[{ch['document_title']} | {ch['section']}]\n{ch['text']}"
        emb = compute_embedding(formatted_content)
        
        ids.append(chunk_id)
        documents.append(formatted_content)
        metadatas.append({
            "source": ch["source"],
            "document_title": ch["document_title"],
            "section": ch["section"],
            "char_count": len(ch["text"])
        })
        embeddings.append(emb)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    duration = round(time.time() - start_time, 3)
    return {
        "status": "success",
        "message": f"Successfully ingested {len(all_chunks)} chunks from {len(md_files)} documents.",
        "files_indexed": len(md_files),
        "file_names": [os.path.basename(f) for f in md_files],
        "chunks_indexed": len(all_chunks),
        "duration_seconds": duration,
        "store_path": store_dir
    }


def retrieve_context(
    query: str,
    top_k: int = 3,
    store_dir: str = DEFAULT_STORE_DIR
) -> Dict[str, Any]:
    """
    Retrieves the top-k most relevant chunks for a clinical triage query via cosine similarity.
    Returns chunk contents, metadata, and calculated cosine similarity scores.
    """
    t0 = time.time()
    collection = get_kb_collection(store_dir)

    total_chunks = collection.count()
    if total_chunks == 0:
        return {
            "query": query,
            "chunks": [],
            "retrieval_latency_ms": round((time.time() - t0) * 1000, 2),
            "warning": "Knowledge base vector collection is empty. Run /ingest first."
        }

    actual_k = min(top_k, total_chunks)
    query_emb = compute_embedding(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieval_time = round((time.time() - t0) * 1000, 2)
    retrieved_chunks = []

    if results and "documents" in results and results["documents"]:
        docs = results["documents"][0]
        metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
        dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)

        for doc, meta, dist in zip(docs, metas, dists):
            # With cosine space in chromadb, distance = 1 - cosine_similarity
            cosine_similarity = max(0.0, min(1.0, 1.0 - dist))
            retrieved_chunks.append({
                "source": meta.get("source", "unknown"),
                "document_title": meta.get("document_title", "Unknown"),
                "section": meta.get("section", "General"),
                "text": doc,
                "cosine_distance": round(float(dist), 4),
                "relevance_score": round(cosine_similarity, 4)
            })

    return {
        "query": query,
        "top_k": actual_k,
        "retrieval_latency_ms": retrieval_time,
        "chunks": retrieved_chunks
    }


def get_kb_stats(store_dir: str = DEFAULT_STORE_DIR) -> Dict[str, Any]:
    """Returns statistics about the current persistent vector store."""
    try:
        collection = get_kb_collection(store_dir)
        count = collection.count()
        return {
            "initialized": True,
            "chunk_count": count,
            "collection_name": COLLECTION_NAME,
            "store_path": store_dir
        }
    except Exception as e:
        return {
            "initialized": False,
            "chunk_count": 0,
            "error": str(e)
        }
