import os
import time
import ollama


DOCUMENT_FOLDER = "documents"


# ============================================================
# LOAD DOCUMENTS
# ============================================================

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


# ============================================================
# SEARCH DOCUMENTS
# ============================================================

def search_documents(question, documents):

    question_words = question.lower().split()

    results = []

    for document in documents:

        text = document["text"].lower()

        score = 0

        for word in question_words:

            if len(word) > 3 and word in text:
                score += 1

        if score > 0:

            results.append({
                "filename": document["filename"],
                "text": document["text"],
                "score": score
            })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:5]


# ============================================================
# ASK OLLAMA
# ============================================================

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


    # --------------------------------------------------------
    # START OLLAMA TIMER
    # --------------------------------------------------------

    ollama_start = time.perf_counter()


    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    ollama_end = time.perf_counter()


    # --------------------------------------------------------
    # GET RESPONSE
    # --------------------------------------------------------

    answer = response.message.content


    # --------------------------------------------------------
    # OLLAMA PERFORMANCE INFORMATION
    # --------------------------------------------------------

    ollama_time = ollama_end - ollama_start


    # Ollama statistics
    total_duration = getattr(
        response,
        "total_duration",
        None
    )

    load_duration = getattr(
        response,
        "load_duration",
        None
    )

    prompt_eval_count = getattr(
        response,
        "prompt_eval_count",
        None
    )

    prompt_eval_duration = getattr(
        response,
        "prompt_eval_duration",
        None
    )

    eval_count = getattr(
        response,
        "eval_count",
        None
    )

    eval_duration = getattr(
        response,
        "eval_duration",
        None
    )


    return (
        answer,
        ollama_time,
        total_duration,
        load_duration,
        prompt_eval_count,
        prompt_eval_duration,
        eval_count,
        eval_duration
    )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

documents = load_documents()


print("====================================")
print(" WELLBEING TECHNOLOGY RAG AGENT")
print("====================================")

print(
    f"Loaded {len(documents)} documents."
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    question = input(
        "\nAsk your question (type 'exit' to quit): "
    )


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if question.lower().strip() == "exit":

        print("\nGoodbye! 👋")

        break


    # --------------------------------------------------------
    # EMPTY QUESTION
    # --------------------------------------------------------

    if not question.strip():

        print("Please enter a question.")

        continue


    # ========================================================
    # TOTAL QUERY TIMER START
    # ========================================================

    total_start = time.perf_counter()


    # ========================================================
    # DOCUMENT RETRIEVAL
    # ========================================================

    retrieval_start = time.perf_counter()


    results = search_documents(
        question,
        documents
    )


    retrieval_end = time.perf_counter()


    retrieval_time = (
        retrieval_end - retrieval_start
    )


    # ========================================================
    # NO RESULTS
    # ========================================================

    if not results:

        total_end = time.perf_counter()

        total_time = (
            total_end - total_start
        )

        print(
            "\nI could not find relevant information."
        )

        print(
            f"Query time: {total_time:.3f} seconds"
        )

        continue


    # ========================================================
    # SHOW RETRIEVED DOCUMENTS
    # ========================================================

    print("\n")
    print("RETRIEVED DOCUMENTS")
    print("====================================")

    for result in results:

        print(
            f"- {result['filename']} "
            f"(score: {result['score']})"
        )


    # ========================================================
    # ASK OLLAMA
    # ========================================================

    (
        answer,
        ollama_time,
        total_duration,
        load_duration,
        prompt_eval_count,
        prompt_eval_duration,
        eval_count,
        eval_duration
    ) = ask_ollama(
        question,
        results
    )


    # ========================================================
    # TOTAL QUERY TIME
    # ========================================================

    total_end = time.perf_counter()

    total_query_time = (
        total_end - total_start
    )


    # ========================================================
    # ANSWER
    # ========================================================

    print("\n")
    print("ANSWER")
    print("====================================")

    print(answer)


    # ========================================================
    # SOURCES
    # ========================================================

    print("\n")
    print("SOURCES")
    print("====================================")

    for result in results:

        print(
            f"- {result['filename']}"
        )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    print("\n")
    print("PERFORMANCE")
    print("====================================")


    print(
        f"Total RAG query time: "
        f"{total_query_time:.3f} seconds"
    )


    print(
        f"Document retrieval time: "
        f"{retrieval_time:.3f} seconds"
    )


    print(
        f"Ollama Python call time: "
        f"{ollama_time:.3f} seconds"
    )


    # --------------------------------------------------------
    # OLLAMA STATISTICS
    # --------------------------------------------------------

    if total_duration is not None:

        print(
            f"Ollama total duration: "
            f"{total_duration / 1_000_000_000:.3f} seconds"
        )


    if load_duration is not None:

        print(
            f"Model load time: "
            f"{load_duration / 1_000_000_000:.3f} seconds"
        )


    if prompt_eval_count is not None:

        print(
            f"Prompt tokens: "
            f"{prompt_eval_count}"
        )


    if eval_count is not None:

        print(
            f"Generated tokens: "
            f"{eval_count}"
        )


    # --------------------------------------------------------
    # TOKEN SPEED
    # --------------------------------------------------------

    if eval_count is not None and eval_duration:

        tokens_per_second = (
            eval_count /
            (eval_duration / 1_000_000_000)
        )

        print(
            f"Generation speed: "
            f"{tokens_per_second:.2f} tokens/second"
        )


    if prompt_eval_count is not None and prompt_eval_duration:

        prompt_tokens_per_second = (
            prompt_eval_count /
            (prompt_eval_duration / 1_000_000_000)
        )

        print(
            f"Prompt processing speed: "
            f"{prompt_tokens_per_second:.2f} tokens/second"
        )


    print("====================================")
