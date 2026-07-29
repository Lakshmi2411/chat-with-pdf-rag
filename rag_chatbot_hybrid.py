import os
import requests
from dotenv import load_dotenv
from pdf_loader import load_and_chunk_pdf
from sparse_retrieval import build_bm25_index
from hybrid_retrieval import hybrid_search

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
URL = "https://api.groq.com/openai/v1/chat/completions"

# Reusing the one-shot prompt style (rules + one worked example + citation
# requirement), since that gave us the most consistent, well-formatted
# answers earlier. The only thing changing in this file is the RETRIEVAL
# method (hybrid instead of dense-only), the prompting layer is independent
# of the retrieval layer, that separation is intentional and worth noting.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context from a document.

Rules:
- Only use information found in the context below to answer.
- If the answer is not contained in the context, say "I could not find that information
  in the document." Do not make up an answer.
- Be concise and direct.
- Always cite the section name the answer came from, in brackets at the end.

Example:
Context: "5. Remote Work Equipment: NimbusTech provides a one-time home office setup
allowance of £300 to all remote and hybrid employees."
Question: "What is the home office allowance?"
Answer: "The home office setup allowance is £300. [Remote Work Equipment]"
"""


def build_prompt(question, context_chunks):
    context_text = "\n\n---\n\n".join(context_chunks)
    return f"""Context from document:
{context_text}

Question: {question}

Answer the question using only the context above."""


def ask_question(question, chunks, bm25_index, top_k=3, temperature=0.2):
    # Step 1: Retrieve — hybrid = dense (ChromaDB) + sparse (BM25), fused via RRF
    context_chunks, _ = hybrid_search(question, chunks, bm25_index, top_k=top_k)

    # Step 2: Augment
    user_prompt = build_prompt(question, context_chunks)

    # Step 3: Generate
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    response = requests.post(URL, headers=headers, json=payload)
    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    return answer, context_chunks


if __name__ == "__main__":
    print("Chat with your PDF — Hybrid Retrieval mode (type 'exit' to quit)\n")

    pdf_path = input("Enter path to a PDF file: ")
    # NOTE: this assumes you've already run vector_store.py once on this PDF,
    # so the dense embeddings are already stored in ChromaDB. This script only
    # rebuilds the BM25 (sparse) index fresh, since that one doesn't persist.
    chunks = load_and_chunk_pdf(pdf_path, chunk_size=150, overlap=30)
    bm25_index = build_bm25_index(chunks)

    while True:
        question = input("\nYou: ")
        if question.lower() == "exit":
            break

        answer, used_chunks = ask_question(question, chunks, bm25_index)

        print(f"\nBot: {answer}\n")

        show_sources = input("Show retrieved chunks? (y/n): ").lower()
        if show_sources == "y":
            for i, chunk in enumerate(used_chunks, 1):
                print(f"\n[Source {i}]\n{chunk}")
