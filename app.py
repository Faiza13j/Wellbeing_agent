import os
import re

# Folder containing our text files
DATA_FOLDER = "documents"


def search_files(question):
    results = []

    # Convert the question to lowercase
    question = question.lower()

    # Get words from the question
    words = re.findall(r'\b[a-zA-Z]+\b', question)

    # Ignore common words
    stop_words = [
        "the", "is", "a", "an", "and", "or", "for",
        "to", "of", "in", "on", "with", "how",
        "what", "which", "can", "does", "do",
        "helps", "help", "my"
    ]

    # Keep only important words
    keywords = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    # Search every text file
    for filename in os.listdir(DATA_FOLDER):

        if filename.endswith(".txt"):

            filepath = os.path.join(DATA_FOLDER, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                content = file.read()

            content_lower = content.lower()

            # Count matching keywords
            score = 0

            for word in keywords:
                if word in content_lower:
                    score += 1

            # Save files with matching words
            if score > 0:
                results.append((score, filename, content))

    # Show the best matching file first
    results.sort(reverse=True)

    return results


# -----------------------------
# MAIN PROGRAM
# -----------------------------

print("=" * 55)
print("     WELLBEING TECHNOLOGY INFORMATION AGENT")
print("=" * 55)

print("Ask a question about wellbeing technology.")
print("Type 'exit' to close the program.")
print()


while True:

    question = input("🔎 Ask your question: ")

    # Exit
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Empty question
    if question.strip() == "":
        print("Please enter a question.")
        continue

    # Search documents
    results = search_files(question)

    print()

    if results:

        print(f"Found {len(results)} relevant document(s)")
        print("=" * 55)

        # Show the top 3 matching documents
        for score, filename, content in results[:3]:

            print()
            print(f"📄 Document: {filename}")
            print(f"⭐ Match score: {score}")
            print("-" * 55)
            print(content)

    else:

        print("❌ No relevant information found.")
        print("Try asking in a different way.")

    print()
    print("=" * 55)
    print()