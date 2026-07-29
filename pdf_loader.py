import pypdf


def extract_text_from_pdf(file_path):
    """Extracts raw text from a PDF, page by page."""
    reader = pypdf.PdfReader(file_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Splits text into overlapping chunks by word count.

    chunk_size: number of words per chunk
    overlap: number of words repeated between consecutive chunks, so context
             isn't lost right at chunk boundaries
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += chunk_size - overlap  # move forward, but re-include the overlap

    return chunks


def load_and_chunk_pdf(file_path, chunk_size=500, overlap=50):
    """Convenience function: extract + chunk in one call."""
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text, chunk_size, overlap)
    return chunks


if __name__ == "__main__":
    pdf_path = input("Enter path to a PDF file: ")
    chunks = load_and_chunk_pdf(pdf_path)

    print(f"\nExtracted {len(chunks)} chunks from the PDF.\n")
    print("--- First chunk preview ---\n")
    print(chunks[0][:500] + "...")
