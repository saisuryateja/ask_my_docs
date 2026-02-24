import nltk
from nltk.tokenize import sent_tokenize

# Ensure required NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def choose_chunk_params(total_words: int):
    """
    Decide chunk size and overlap based on document length
    optimized for all-MiniLM-L6-v2 (256 token limit)
    """
    if total_words < 1_000:
        return 100, 2          # small docs
    else:
        return 180, 2          # medium/large docs (optimized for speed)

def chunk_text(pages: list[dict], max_words: int, overlap_sentences: int):
    """
    Chunks text while keeping track of page numbers.
    Returns a list of dicts: {"text": str, "pages": list[int]}
    """
    all_chunks = []
    
    for page_data in pages:
        page_num = page_data["page"]
        text = page_data["text"]
        sentences = sent_tokenize(text)
        
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            word_count = len(words)
            
            if current_word_count + word_count > max_words and current_chunk:
                all_chunks.append({
                    "text": " ".join(current_chunk),
                    "pages": [page_num]
                })
                
                # Simple overlap within the page
                overlap = (
                    current_chunk[-overlap_sentences:]
                    if overlap_sentences < len(current_chunk)
                    else current_chunk
                )
                current_chunk = overlap.copy()
                current_word_count = sum(len(s.split()) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_word_count += word_count
            
        if current_chunk:
            all_chunks.append({
                "text": " ".join(current_chunk),
                "pages": [page_num]
            })

    return all_chunks
