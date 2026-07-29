# Project Notes: Chat with PDF (RAG)

## 1. Project Overview

A RAG (Retrieval-Augmented Generation) system that answers questions grounded in a specific
PDF document, rather than relying on the LLM's general training knowledge. The project was
deliberately structured to build and compare multiple techniques on the same document, so
you can see *why* each one matters, not just that it "works."

**The core loop every RAG system follows:**
```
Question → Retrieve relevant chunks → Stuff chunks into prompt → LLM generates
           a grounded answer using ONLY that context
```

---

## 2. Key Concepts (Keywords) Explained

### Chunking
Splitting a long document into smaller pieces (chunks) before embedding/searching, because
you can't feasibly search or feed an entire document into an LLM prompt every time. We used
word-count based sliding-window chunking (`chunk_size`, `overlap`), simple, but it means
chunk boundaries don't align with the document's actual section boundaries, a single chunk
can span two unrelated topics.

### Embeddings
A way of converting text into a list of numbers (a vector) that captures its *meaning*.
Texts with similar meaning end up with vectors that point in similar directions, even if
they don't share exact words. This is what let dense retrieval match "home office equipment
allowance" to the document's "home office setup allowance", different words, same meaning.

### Dense Retrieval (Semantic Search)
Retrieval based on embedding similarity (cosine similarity). Good at matching *meaning*
even when wording differs. Implemented in `vector_store.py` using `sentence-transformers`
+ ChromaDB.

### Sparse Retrieval (BM25 / Keyword Search)
Retrieval based on exact word overlap, weighted by how rare/distinctive each word is
across the document set (this weighting is called **IDF**, inverse document frequency).
Doesn't understand meaning or synonyms, but is precise when exact terms matter. Implemented
in `sparse_retrieval.py` using `rank_bm25`.

**Important lesson learned:** naive tokenization (just `.lower().split()`) fails silently.
A query word like `"allowance?"` (with punctuation attached) will never match the document's
`"allowance"`, letting common stopwords like "the" and "is" dominate the ranking instead.
Always strip punctuation before tokenizing for BM25.

### Hybrid Retrieval
Combining dense and sparse retrieval so each covers the other's weaknesses. The tricky part:
dense similarity scores (~0-1) and BM25 scores (unbounded) are on completely different
scales, so you can't just average them. **Reciprocal Rank Fusion (RRF)** solves this by
ignoring raw scores entirely and combining based on each method's *rank* (1st place, 2nd
place, etc):
```
RRF_score(chunk) = sum over each retrieval method of  1 / (k + rank_in_that_method)
```
where `k` (commonly 60) is a smoothing constant. A chunk that ranks well in *both* methods
gets a strong combined score.

### Zero-Shot Prompting
Giving the model instructions only, no example of what a good answer looks like. Reliable
for simple, single-fact lookups.

### One-Shot Prompting
Instructions **plus one worked example** showing the exact input → output pattern desired
(in this project: a fact + a `[Section Name]` citation). Doesn't make answers more
*correct*, dense retrieval already handled correctness, it makes answers more *consistent
and structured*.

### Chain-of-Thought (CoT) Prompting
Explicitly instructing the model to reason step by step (shown under a `Reasoning:` label)
before giving a final answer. Necessary for **multi-hop questions** that require combining
facts from more than one section (e.g. "does my probation status affect my L&D budget
access?", which requires checking two separate, unrelated sections and confirming they
don't interact). Zero-shot and one-shot both jump straight to a conclusion; CoT forces the
model to show its work, which also makes it possible to *check* whether the reasoning
was actually sound.

### Temperature (revisited)
Even a "low" temperature like `0.2` still introduces some randomness. We saw this directly:
the exact same question, asked twice in a row, produced two slightly different (though both
correct) answers. For deterministic, factual Q&A, `temperature=0` is the right choice, not
just a low number.

### Retrieval Noise
Because `top_k` always returns a fixed number of chunks, some of the returned chunks are
often irrelevant, they get pulled in because *something* had to fill the remaining slots,
not because they're a good match. This is normal and something a well-prompted LLM can
generally reason past ("only use information found in the context above"), but it's worth
noticing rather than assuming retrieval failed.

---

## 3. Important Syntax Explained

### Sliding-window chunking
```python
start = 0
while start < len(words):
    end = start + chunk_size
    chunks.append(" ".join(words[start:end]))
    start += chunk_size - overlap   # overlap prevents context loss at chunk boundaries
```

### Storing embeddings in ChromaDB
```python
collection.add(
    documents=chunks,       # original text, so it can be returned later
    embeddings=embeddings,  # the vector representation
    ids=ids                 # unique identifier per chunk
)
```

### Querying ChromaDB for similar chunks
```python
results = collection.query(query_embeddings=query_embedding, n_results=top_k)
top_chunks = results["documents"][0]
```

### Building and querying a BM25 index
```python
tokenized_chunks = [tokenize(chunk) for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)
scores = bm25.get_scores(tokenized_query)
```

### Proper tokenization for BM25 (the fix)
```python
import re
def tokenize(text):
    text = re.sub(r"[^\w\s]", "", text.lower())  # strip punctuation first
    return text.split()
```

### Reciprocal Rank Fusion
```python
scores = {}
for rank, chunk in enumerate(dense_chunks, start=1):
    scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank)
for rank, chunk in enumerate(sparse_chunks, start=1):
    scores[chunk] = scores.get(chunk, 0) + 1 / (k + rank)
ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
```
`enumerate(..., start=1)` gives each chunk its rank position (1st, 2nd, 3rd...) alongside
the chunk itself, which is exactly what the RRF formula needs, not the raw scores.

### Building the RAG prompt (context stuffing)
```python
context_text = "\n\n---\n\n".join(context_chunks)
user_prompt = f"""Context from document:
{context_text}

Question: {question}

Answer the question using only the context above."""
```
This is the "Augment" step of RAG, retrieved chunks get joined into one text block and
inserted directly into the prompt sent to the LLM.

---

## 4. Debugging Lessons Worth Remembering

1. **Truncated debug previews can mislead you.** Early on, printing only the first 200-300
   characters of a chunk made it look like retrieval had "missed" content, when the full
   chunk (sent to the LLM in full) actually contained the answer further down. Always print
   full text when debugging retrieval, not a preview.
2. **A word-count based chunker will regularly straddle two unrelated sections.** This is
   normal, but worth checking for directly (e.g. printing full chunks with word counts)
   rather than assuming.
3. **Test retrieval methods with deliberately paraphrased queries.** Using the document's
   exact wording as your test question doesn't tell you much, every method would find it.
   Rephrasing the question so it shares few literal words with the source text is what
   actually reveals the difference between dense (meaning-based) and sparse (word-based)
   retrieval.

---

## 5. The Bigger Picture

This project reinforced that a "RAG system" isn't one algorithm, it's a **pipeline of
independent, swappable stages**:
```
Chunking → Retrieval (dense / sparse / hybrid) → Prompting (zero-shot / one-shot / CoT) → Generation
```
Each stage can be improved or swapped independently without touching the others, this is
exactly why `rag_chatbot_hybrid.py` and `rag_chatbot_cot.py` could reuse the same
`build_prompt()` pattern while changing only the retrieval method, and why `vector_store.py`
never needed to know anything about prompting at all. Understanding this separation is
probably the single most transferable insight from this project, it's how virtually all
production RAG and agentic systems are actually structured.