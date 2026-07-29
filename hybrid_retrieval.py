from vector_store import search_similar_chunks
from sparse_retrieval import build_bm25_index, search_bm25

# --- HYBRID RETRIEVAL (Dense + Sparse, combined via Reciprocal Rank Fusion) ---
#
# Why not just average the two scores together? Because dense similarity
# scores (cosine similarity, roughly 0-1) and BM25 scores (unbounded, can be
# 0, 2, 5, 15...) live on completely different scales. Averaging them
# directly would let whichever method happens to produce bigger numbers
# dominate, that's not a fair combination.
#
# Reciprocal Rank Fusion (RRF) sidesteps this entirely by ignoring the raw
# scores and using only each method's RANKING (1st place, 2nd place, etc).
# A chunk that ranks highly in BOTH methods gets a strong combined score,
# even if the two methods scored it very differently in absolute terms.


def reciprocal_rank_fusion(dense_chunks, sparse_chunks, k=60):
    """
    dense_chunks and sparse_chunks are both ORDERED lists of chunk text,
    best match first (rank 1), from each retrieval method.

    RRF formula for each chunk: score = sum of 1 / (k + rank) across every
    method it appears in. 'k' is a smoothing constant (60 is the standard
    default from the original RRF paper), it stops rank #1 from
    completely dominating rank #2.
    """
    scores = {}

    for rank, chunk in enumerate(dense_chunks, start=1):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank)

    for rank, chunk in enumerate(sparse_chunks, start=1):
        scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank)

    # Sort chunks by their combined RRF score, highest first
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    return ranked


def hybrid_search(query, chunks, bm25_index, pool_size=5, top_k=3):
    """
    Runs both retrieval methods, pulling a slightly larger 'pool_size' from
    each (not just top_k), so RRF has more candidates to fairly re-rank
    before we cut down to the final top_k.
    """
    dense_results = search_similar_chunks(query, top_k=pool_size)
    sparse_results, _ = search_bm25(query, chunks, bm25_index, top_k=pool_size)

    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    top_chunks = [chunk for chunk, score in fused[:top_k]]
    top_fused_scores = [score for chunk, score in fused[:top_k]]

    return top_chunks, top_fused_scores


if __name__ == "__main__":
    from pdf_loader import load_and_chunk_pdf

    pdf_path = input("Enter path to a PDF file: ")
    chunks = load_and_chunk_pdf(pdf_path, chunk_size=150, overlap=30)

    # Dense chunks are assumed already stored via vector_store.py's store_chunks().
    # Sparse (BM25) index is rebuilt fresh here, it's cheap and doesn't persist to disk.
    bm25_index = build_bm25_index(chunks)

    query = input("\nEnter a test question: ")
    top_chunks, top_scores = hybrid_search(query, chunks, bm25_index)

    print(
        f"\n--- Top {len(top_chunks)} matching chunks (Hybrid: Dense + BM25 via RRF) ---\n"
    )
    for i, (chunk, score) in enumerate(zip(top_chunks, top_scores), 1):
        print(f"[{i}] (RRF score: {score:.4f}) {chunk[:300]}...\n")
