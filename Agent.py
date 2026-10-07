import os
import ollama


DOCUMENT_FOLDER = "documents"


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

    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


documents = load_documents()

print("====================================")
print(" WELLBEING TECHNOLOGY AGENT")
print("====================================")

while True:

    question = input("\nAsk your question (type 'exit' to quit): ")

    # Exit the agent
    if question.lower().strip() == "exit":
        print("\nGoodbye! 👋")
        break

    # Ignore empty questions
    if not question.strip():
        print("Please enter a question.")
        continue

    # Search documents
    results = search_documents(
        question,
        documents
    )

    if not results:

        print("\nI could not find relevant information.")

        continue

    # Ask Ollama
    answer = ask_ollama(
        question,
        results
    )

    print("\n")
    print("ANSWER")
    print("====================================")
    print(answer)

    print("\nSOURCES")
    print("====================================")

    for result in results:

        print(
            f"- {result['filename']}"
        )