import os
import requests
from dotenv import load_dotenv
from pdf_loader import load_and_chunk_pdf
from sparse_retrieval import build_bm25_index
from hybrid_retrieval import hybrid_search

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
URL = "https://api.groq.com/openai/v1/chat/completions"

# --- CHAIN-OF-THOUGHT (CoT) SYSTEM PROMPT ---
# Zero-shot and one-shot both jump straight to a final answer. That's fine
# for single-fact lookups ("what is X"), but it breaks down on questions
# that need REASONING ACROSS MULTIPLE FACTS, e.g. combining a rule from one
# section with a rule from another to work out something the document never
# states directly.
#
# Chain-of-thought fixes this by explicitly instructing the model to work
# through the relevant facts step by step BEFORE giving a final answer,
# rather than jumping straight to a conclusion. We ask for the reasoning to
# be shown (under "Reasoning:") separately from the final answer (under
# "Answer:"), so you can inspect whether the model's logic was actually
# sound, not just whether the final answer sounds right.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context from a document.

Rules:
- Only use information found in the context below to answer.
- If the answer is not contained in the context, say "I could not find that information
  in the document." Do not make up an answer.
- For questions that require combining multiple facts from the context, think through
  the relevant facts step by step BEFORE giving your final answer.
- Always structure your response in exactly this format:

Reasoning: <your step-by-step reasoning here, referencing the specific facts you're using>
Answer: <your final, concise answer, with a section citation in brackets>

Example:
Context: "3. Annual Leave Policy: Full-time employees receive 25 days of annual leave.
Unused leave of up to 5 days may be carried over into the following year, but must be
used by March 31st or it will be forfeited."
Question: "If I have 8 unused leave days at the end of the year, how many will I lose?"
Reasoning: The policy allows up to 5 unused days to be carried over. The employee has
8 unused days. That means 8 - 5 = 3 days exceed the carryover limit and will be forfeited.
Answer: You would lose 3 days, since only 5 unused days can be carried over. [Annual Leave Policy]
"""


def build_prompt(question, context_chunks):
    context_text = "\n\n---\n\n".join(context_chunks)
    return f"""Context from document:
{context_text}

Question: {question}

Think through this step by step using only the context above, then give your final answer
in the required Reasoning / Answer format."""


def ask_question(question, chunks, bm25_index, top_k=4, temperature=0.2):
    # Multi-hop questions often need facts from more than one section, so we
    # retrieve a slightly larger top_k here (4) than the single-fact chatbots (3).
    context_chunks, _ = hybrid_search(question, chunks, bm25_index, top_k=top_k)

    user_prompt = build_prompt(question, context_chunks)

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
    print("Chat with your PDF — Chain-of-Thought mode (type 'exit' to quit)\n")

    pdf_path = input("Enter path to a PDF file: ")
    chunks = load_and_chunk_pdf(pdf_path, chunk_size=150, overlap=30)
    bm25_index = build_bm25_index(chunks)

    while True:
        question = input("\nYou: ")
        if question.lower() == "exit":
            break

        answer, used_chunks = ask_question(question, chunks, bm25_index)

        print(f"\nBot:\n{answer}\n")

        show_sources = input("Show retrieved chunks? (y/n): ").lower()
        if show_sources == "y":
            for i, chunk in enumerate(used_chunks, 1):
                print(f"\n[Source {i}]\n{chunk}")
