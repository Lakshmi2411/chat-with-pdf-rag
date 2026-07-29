import os
import requests
from dotenv import load_dotenv
from vector_store import search_similar_chunks

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
URL = "https://api.groq.com/openai/v1/chat/completions"

# --- ZERO-SHOT SYSTEM PROMPT ---
# "Zero-shot" means we give the model instructions but NO worked example of
# a question/answer pair. It has to follow the instructions cold.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context from a document.

Rules:
- Only use information found in the context below to answer.
- If the answer is not contained in the context, say "I could not find that information
  in the document." Do not make up an answer.
- Be concise and direct.
"""


def build_prompt(question, context_chunks):
    """
    Combines the retrieved chunks into a single context block, then builds
    the final user prompt that gets sent alongside the system prompt.
    This is the core of RAG: 'stuffing' retrieved context into the prompt
    so the model can ground its answer in it, rather than relying on what
    it memorized during training.
    """
    context_text = "\n\n---\n\n".join(context_chunks)

    user_prompt = f"""Context from document:
{context_text}

Question: {question}

Answer the question using only the context above."""

    return user_prompt


def ask_question(question, top_k=3, temperature=0.2):
    # Step 1: Retrieve — find the most relevant chunks for this question
    context_chunks = search_similar_chunks(question, top_k=top_k)

    # Step 2: Augment — build a prompt that includes those chunks as context
    user_prompt = build_prompt(question, context_chunks)

    # Step 3: Generate — send it to the LLM and get a grounded answer back
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
    print("Chat with your PDF (type 'exit' to quit)\n")

    while True:
        question = input("You: ")
        if question.lower() == "exit":
            break

        answer, used_chunks = ask_question(question)

        print(f"\nBot: {answer}\n")

        # Optional: show which chunks were used, useful for debugging retrieval
        show_sources = input("Show retrieved chunks? (y/n): ").lower()
        if show_sources == "y":
            for i, chunk in enumerate(used_chunks, 1):
                print(f"\n[Source {i}] {chunk[:200]}...")
        print()
