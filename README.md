# Chat with PDF — RAG (Retrieval-Augmented Generation)

A terminal-based Q&A system that lets you "chat" with any PDF document. Built as a learning
project to understand and compare the core building blocks of real-world RAG systems:
chunking, multiple retrieval strategies, and multiple prompting styles, all on the same
document, so each technique's effect can be directly compared.

## What This Project Demonstrates

Rather than building a single RAG pipeline, this project deliberately builds and compares
**three retrieval strategies** and **three prompting styles**, so the differences between
them are visible side by side rather than theoretical.

### Retrieval strategies

| Strategy | File | How it works |
|---|---|---|
| Dense (semantic) | `vector_store.py` | Embeds text into vectors using `sentence-transformers`, stores them in ChromaDB, retrieves by cosine similarity |
| Sparse (keyword) | `sparse_retrieval.py` | Classic BM25 keyword ranking, no embeddings involved |
| Hybrid | `hybrid_retrieval.py` | Combines dense + sparse rankings using Reciprocal Rank Fusion (RRF) |

### Prompting styles

| Style | File | What's different |
|---|---|---|
| Zero-shot | `rag_chatbot_zero_shot.py` | Instructions only, no example |
| One-shot | `rag_chatbot_one_shot.py` | Instructions + one worked example, enforces a section-citation format |
| Chain-of-thought | `rag_chatbot_cot.py` | Model must show step-by-step reasoning before answering, needed for multi-hop questions that combine facts from more than one section |

## Tech Stack

- Python 3.13
- Groq API (Llama 3.3 70B) for generation
- `sentence-transformers` (`all-MiniLM-L6-v2`) for local, free embeddings
- ChromaDB for persistent vector storage
- `rank_bm25` for keyword-based sparse retrieval
- `pypdf` for PDF text extraction
- `python-dotenv` for environment variable management

## Project Structure

```
chat-with-pdf-rag/
├── pdf_loader.py              # Extracts and chunks PDF text
├── vector_store.py            # Dense retrieval (embeddings + ChromaDB)
├── sparse_retrieval.py        # Sparse retrieval (BM25 keyword search)
├── hybrid_retrieval.py        # Hybrid retrieval (RRF fusion of both)
├── rag_chatbot_zero_shot.py   # Zero-shot prompting + dense retrieval
├── rag_chatbot_one_shot.py    # One-shot prompting + dense retrieval
├── rag_chatbot_hybrid.py      # One-shot prompting + hybrid retrieval
├── rag_chatbot_cot.py         # Chain-of-thought prompting + hybrid retrieval
├── sample_pdfs/
│   └── sample_company_handbook.pdf
└── requirements.txt
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Lakshmi2411/chat-with-pdf-rag.git
   cd chat-with-pdf-rag
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv ai-env
   ai-env\Scripts\activate    # Windows
   source ai-env/bin/activate # Mac/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory:
   ```
   GROQ_API_KEY=your_key_here
   ```
   (Get a free key from [console.groq.com](https://console.groq.com))

## Usage

**1. Embed the PDF (only needs to be run once per document):**
```bash
python vector_store.py
```
This extracts, chunks, and stores the PDF's dense embeddings in a local ChromaDB
(`./chroma_db`), which persists across runs.

**2. Chat with the PDF, using any of the four chatbot variants:**
```bash
python rag_chatbot_zero_shot.py
python rag_chatbot_one_shot.py
python rag_chatbot_hybrid.py
python rag_chatbot_cot.py
```
Each will ask for the PDF path (needed to rebuild the in-memory BM25 index, which doesn't
persist to disk the way ChromaDB does), then let you ask questions interactively. Type
`exit` to quit. After each answer, you can optionally view the full retrieved chunks used
to generate it.

## Example Output

**One-shot (with citation):**
```
You: What is the home office equipment allowance?
Bot: The home office setup allowance is £300. [Remote Work Equipment]
```

**Chain-of-thought (multi-hop reasoning):**
```
You: If I joined the company today and want to take a course next month, would my
     probation status affect my ability to use the learning and development budget?

Bot:
Reasoning: The L&D budget rule doesn't mention any tenure requirement. The probation
period rule (3 months) doesn't mention any budget restriction either. There is a
separate 12-month tenure rule, but it applies to enhanced parental leave, a different
benefit entirely, so it doesn't apply here.
Answer: No, your probation status would not affect your ability to use the learning
and development budget. [Learning and Development Budget]
```

## How It Works

Every chatbot variant follows the same core RAG loop:

**Retrieve → Augment → Generate**
1. **Retrieve**: the user's question is embedded (and/or keyword-matched) to find the
   most relevant chunks from the document
2. **Augment**: those chunks are inserted into the prompt as context
3. **Generate**: the LLM answers using *only* that context, grounding its response in
   the actual document rather than its own general knowledge

Retrieval and prompting are deliberately separated: swapping `search_similar_chunks()`
for `hybrid_search()` doesn't require changing the system prompt at all, and vice versa.
This mirrors how production RAG systems are architected.

## Known Limitations / Design Notes

- **Chunking is word-count based**, not structure-aware. A single chunk can span across
  two unrelated sections (e.g. "Annual Leave" and "Probation Period"), which occasionally
  introduces retrieval noise. A v2 improvement would be structure-aware chunking (splitting
  on section headers first, then applying size limits within each section).
- **BM25 needs proper tokenization.** An early version of `sparse_retrieval.py` split text
  on whitespace only, which meant punctuation-attached words (e.g. `"allowance?"`) never
  matched the document's `"allowance"`, letting stopword frequency dominate the ranking
  instead. Fixed by stripping punctuation before tokenizing.
- **Temperature matters even at low values.** Identical questions asked twice at
  `temperature=0.2` produced two slightly different (though both accurate) answers.
  Fully deterministic factual Q&A should use `temperature=0`.
- **Small corpora make BM25/IDF statistics noisy.** With only ~5 chunks, BM25's rarity
  weighting has very little data to work with, this is a known limitation of keyword-based
  retrieval on small document sets, and part of the motivation for hybrid search.
- No persistence of chat history between runs, each session is independent.

## Future Improvements

- [ ] Multi-query retrieval (generate several reworded versions of the question to
      improve recall)
- [ ] Structure-aware chunking (split on section headers before applying size limits)
- [ ] Similarity score thresholding, to drop clearly irrelevant chunks instead of always
      returning a fixed top-k
- [ ] Streamlit web interface
- [ ] Simple retrieval evaluation (e.g. RAGAS) to measure retrieval quality quantitatively
- [ ] Support for multiple PDFs / a document library, not just one file at a time

## License

MIT