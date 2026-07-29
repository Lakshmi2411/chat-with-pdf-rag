import re
from rank_bm25 import BM25Okapi

# --- SPARSE / KEYWORD RETRIEVAL (BM25) ---
# Unlike dense retrieval (which compares meaning via embeddings), BM25 scores
# chunks based on exact word overlap with the query, weighted so that rare,
# distinctive words count for more than common ones (like "the" or "and").
# It's the classic algorithm behind traditional keyword search engines.


def tokenize(text):
    """
    Lowercases and splits text into words, stripping punctuation first.
    Without this, a query word like "allowance?" would never match the
    document's "allowance", since they'd be treated as completely
    different tokens. This one bug can silently let stopwords (the, is, a)
    dominate the ranking instead of the words that actually matter.
    """
    text = re.sub(r"[^\w\s]", "", text.lower())
    return text.split()


def build_bm25_index(chunks):
    """
    Builds a BM25 index over a list of text chunks.
    BM25Okapi needs the chunks pre-tokenized (split into lists of words),
    not raw strings.
    """
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    return bm25


def search_bm25(query, chunks, bm25_index, top_k=3):
    """
    Scores every chunk against the query using BM25, then returns the
    top_k highest-scoring chunks (original text, not tokens).
    """
    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)

    # Pair each chunk with its score, sort by score descending, take top_k
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda pair: pair[1], reverse=True)

    top_chunks = [chunk for chunk, score in scored_chunks[:top_k]]
    top_scores = [score for chunk, score in scored_chunks[:top_k]]

    return top_chunks, top_scores


if __name__ == "__main__":
    from pdf_loader import load_and_chunk_pdf

    pdf_path = input("Enter path to a PDF file: ")
    chunks = load_and_chunk_pdf(pdf_path, chunk_size=150, overlap=30)

    bm25_index = build_bm25_index(chunks)

    query = input("\nEnter a test question: ")
    top_chunks, top_scores = search_bm25(query, chunks, bm25_index)

    print(f"\n--- Top {len(top_chunks)} matching chunks (BM25) ---\n")
    for i, (chunk, score) in enumerate(zip(top_chunks, top_scores), 1):
        print(f"[{i}] (score: {score:.2f}) {chunk[:300]}...\n")
