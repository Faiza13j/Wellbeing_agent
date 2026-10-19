
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import ollama
import time
import numpy as np
from sentence_transformers import SentenceTransformer


DOCUMENT_FOLDER = "documents"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Number of chunks to retrieve
TOP_K = 3


# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

def load_documents():

    documents = []

    for filename in os.listdir(DOCUMENT_FOLDER):

        if filename.endswith(".txt"):

            filepath = os.path.join(
                DOCUMENT_FOLDER,
                filename
            )

            with open(
                filepath,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()

            documents.append({
                "filename": filename,
                "text": text
            })

    return documents


# --------------------------------------------------
# SPLIT DOCUMENT INTO CHUNKS
# --------------------------------------------------

def create_chunks(documents, chunk_size=300):

    chunks = []

    for document in documents:

        text = document["text"]

        # Split text into words
        words = text.split()

        # Create smaller pieces
        for i in range(0, len(words), chunk_size):

            chunk = " ".join(
                words[i:i + chunk_size]
            )

            if chunk.strip():

                chunks.append({
                    "filename": document["filename"],
                    "text": chunk
                })

    return chunks


# --------------------------------------------------
# CREATE EMBEDDINGS
# --------------------------------------------------

def create_embeddings(chunks, embedding_model):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings




# ----------------------------------------------
# VECTOR SEARCH
# ----------------------------------------------

def search_documents(
    question,
    chunks,
    embeddings,
    embedding_model
):

    # Create a normalized vector for the question
    question_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Calculate similarity with ALL document vectors at once
    scores = embeddings @ question_embedding

    # Get the indices of the highest similarity scores
    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = []

    for i in top_indices:
        results.append({
            "filename": chunks[i]["filename"],
            "text": chunks[i]["text"],
            "score": float(scores[i])
        })

    return results

# --------------------------------------------------
# ASK OLLAMA
# --------------------------------------------------

def ask_ollama(question, results):

    context = ""

    for result in results:

        context += f"""

SOURCE: {result['filename']}

{result['text']}

--------------------------------
"""

    prompt = f"""
You are a wellbeing technology information assistant.

Your job is to answer the user's question using ONLY
the information contained in the provided documents.

Do not invent information.

If the documents do not contain enough information,
say:

"I could not find enough information about this in
the available documents."

USER QUESTION:
{question}

AVAILABLE DOCUMENT INFORMATION:
{context}

Instructions:

1. Answer the question directly.
2. Use simple language.
3. Recommend devices only when the documents support
   the recommendation.
4. Explain briefly why a device is relevant.
5. At the end, list the source filename.
6. Do not make medical diagnoses.
7. Do not claim that a device is medically suitable
   for a specific person.

Answer:
"""

    # Start timer
    start_time = time.time()

    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        think=False,
        keep_alive=-1,
        options={
            "num_predict": 200
        }
    )

    # Stop timer
    response_time = time.time() - start_time

    # Extract final answer
    answer_text = (response.message.content or "").strip()

    # Read token statistics
    prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
    output_tokens = getattr(response, "eval_count", 0) or 0

    print("\nPERFORMANCE")
    print("====================================")
    print(f"Prompt tokens:  {prompt_tokens}")
    print(f"Output tokens:  {output_tokens}")
    print(f"Total tokens:   {prompt_tokens + output_tokens}")
    print(f"Response time:  {response_time:.2f} seconds")

    if not answer_text:
        thinking = getattr(response.message, "thinking", "") or ""
        print("WARNING: Empty answer content.")
        print(f"Thinking field length: {len(thinking)} characters")
        return "No final answer was returned. Check the Ollama output in the terminal."

    return answer_text
# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
     local_files_only=True
)

print("Embedding model loaded.")


# --------------------------------------------------
# LOAD DOCUMENTS
# --------------------------------------------------

documents = load_documents()

print(
    f"Loaded {len(documents)} documents."
)


# --------------------------------------------------
# CREATE CHUNKS
# --------------------------------------------------

chunks = create_chunks(
    documents
)

print(
    f"Created {len(chunks)} document chunks."
)


# --------------------------------------------------
# CREATE DOCUMENT VECTORS
# --------------------------------------------------

print("Creating document embeddings...")

embeddings = create_embeddings(
    chunks,
    embedding_model
)

print("Document embeddings created.")


# --------------------------------------------------
# START AGENT
# --------------------------------------------------
if __name__ == "__main__":

         print("====================================")
         print(" WELLBEING TECHNOLOGY AGENT")
         print("====================================")

print("Semantic vector search is enabled.")

   

while True:

    question = input(
        "\nAsk your question (type 'exit' to quit): "
    )


    # ----------------------------------------------
    # EXIT
    # ----------------------------------------------

    if question.lower().strip() == "exit":

        print("\nGoodbye! 👋")

        break


    # ----------------------------------------------
    # EMPTY QUESTION
    # ----------------------------------------------

    if not question.strip():

        print("Please enter a question.")

        continue


    # ----------------------------------------------
    # VECTOR SEARCH
    # ----------------------------------------------

    results = search_documents(
        question,
        chunks,
        embeddings,
        embedding_model
    )


    # ----------------------------------------------
    # CHECK RESULTS
    # ----------------------------------------------

    if not results:

        print(
            "\nI could not find relevant information."
        )

        continue


    # ----------------------------------------------
    # ASK OLLAMA
    # ----------------------------------------------

    answer = ask_ollama(
        question,
        results
    )


    # ----------------------------------------------
    # DISPLAY ANSWER
    # ----------------------------------------------

    print("\n")
    print("ANSWER")
    print("====================================")

    print(answer)


    # ----------------------------------------------
    # DISPLAY SOURCES
    # ----------------------------------------------

    print("\nSOURCES")
    print("====================================")

    for result in results:

        print(
            f"- {result['filename']}"
        )

    print("\nSIMILARITY SCORES")
    print("====================================")

    for result in results:

        print(
            f"- {result['filename']}: "
            f"{result['score']:.3f}"
        )
