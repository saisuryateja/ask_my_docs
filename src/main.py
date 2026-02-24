from pdf_loader import load_pdf_text
from chunker import chunk_text
from chunker import choose_chunk_params
from embedder import embed_chunks, embed_query, rerank_chunks
from vector_store import VectorStore
import persistence as pr
import xml.etree.ElementTree as ET
from answer_generation import generate_answer
from pathlib import Path
from confidence import is_confident
try:
    import msvcrt
    HAS_MSVCRT = True
except ImportError:
    HAS_MSVCRT = False

BASE_DIR = Path(__file__).parent.parent
def get_pdf_path(data_dir: Path) -> Path:
    """
    Finds the first PDF file in the data directory.
    """
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        return None
    return pdf_files[0]

pdf_path = get_pdf_path(BASE_DIR / 'data')
config_path = BASE_DIR / 'config' / 'param.xml'
chunks_path = BASE_DIR / "config" / "chunks.pkl"
embeddings_path = BASE_DIR / "config" / "embeddings.npy"
faiss_path = BASE_DIR / "config" / "faiss.index"

def config_loader(path: Path):
    """
    Parses the XML configuration file.
    """
    tree = ET.parse(path)

    candidate_k = int(tree.find('candidate_k').text)
    final_k = int(tree.find('final_k').text)
    new_pdf = tree.find('new_pdf').text.lower() == 'true'
    max_tokens = int(tree.find('max_tokens').text)

    return candidate_k, final_k, new_pdf, max_tokens

def process_document(pdf_path: Path, faiss_path: Path, chunks_path: Path, embeddings_path: Path):
    """
    Handles the full ingestion pipeline: text extraction, chunking, and embedding.
    """
    print(f"Ingesting: {pdf_path.name}...")
    pages = load_pdf_text(str(pdf_path))

    # Calculate total words across all pages
    total_words = sum(len(p["text"].split()) for p in pages)
    max_words, overlap_sentences = choose_chunk_params(total_words)

    chunks = chunk_text(pages, max_words, overlap_sentences)
    print(f"Created {len(chunks)} chunks.")

    # Extract text for embedding
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_chunks(chunk_texts)

    vector_store = VectorStore(embeddings)
    vector_store.save(faiss_path)

    pr.save_chunks(chunks, chunks_path)
    pr.save_embeddings(embeddings, embeddings_path)
    
    return chunks, vector_store

candidate_k, final_k, new_pdf, max_tokens = config_loader(config_path)

if __name__ == "__main__":
    if new_pdf:
        if not pdf_path:
            print("Error: No PDF files found in the 'data' directory. Cannot ingest.")
            exit(1)
        chunks, vector_store = process_document(pdf_path, faiss_path, chunks_path, embeddings_path)
    else:
        chunks = pr.load_chunks(chunks_path)
        print(f"Loaded {len(chunks)} existing chunks.")
        vector_store = VectorStore.load(faiss_path)

    while True:
        user_question = input("\nAsk a question (type 'exit' to quit): ").strip()
        if user_question.lower() == "exit":
            break

        retrieval_query = user_question
        query_embedding = embed_query(retrieval_query)
        top_indices, distances = vector_store.search(query_embedding, top_k=candidate_k)

        if not is_confident(distances,abs_threshold=1.5):
            print("\nANSWER:\nAnswer not found in the document.")
            continue

        # Layer 1: Broad Retrieval
        candidate_chunks_data = [chunks[i] for i in top_indices]
        candidate_texts = [c["text"] for c in candidate_chunks_data]
        
        # Layer 2: Re-ranking
        print(f"Reranking top {len(candidate_chunks_data)} candidates...", flush=True)
        # Match texts back to their metadata after re-ranking
        ranked_texts = rerank_chunks(user_question, candidate_texts, top_n=final_k)
        
        # Select the full chunk objects that match the ranked texts
        retrieved_chunks_data = []
        for text in ranked_texts:
            for c in candidate_chunks_data:
                if c["text"] == text:
                    retrieved_chunks_data.append(c)
                    break
        
        if HAS_MSVCRT:
            print("\n(Press 'q' or 'x' to stop generation)\n")
        else:
            print("\nANSWER:\n")

        final_chunk_texts = [c["text"] for c in retrieved_chunks_data]
        full_response = []
        for chunk in generate_answer(final_chunk_texts, user_question,max_tokens):
            if HAS_MSVCRT and msvcrt.kbhit():
                key = msvcrt.getch().decode().lower()
                if key in ['q', 'x']:
                    print("\n\n[Generation Interrupted by User]")
                    break
            print(chunk, end="", flush=True)
            full_response.append(chunk)
        
        # Print Sources
        full_response_text = "".join(full_response).lower
        if "not found" not in full_response_text and pdf_path and retrieved_chunks_data:
            pages = sorted(list(set(p for c in retrieved_chunks_data for p in c["pages"])))
            page_str = ", ".join(map(str, pages))
            print(f"\n\nSOURCES: {pdf_path.name} (Page {page_str})")
        
        print()
        # print(f"[Retrieval] Distances: {distances}")