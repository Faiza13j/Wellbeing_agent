
# Wellbeing Technology AI Agent

**A Retrieval-Augmented Generation (RAG) project for wellbeing technology information**

**Developer:** Faiza Urooj Khan
**Email:** [faizaurooj.khan@edu.turkuamk.fi](mailto:faizaurooj.khan@edu.turkuamk.fi)

## About the Project

The Wellbeing Technology AI Agent is an applied AI prototype that answers questions about wellbeing and safety technologies using information from a collection of text documents.

The project combines semantic search and a locally running large language model (LLM) to retrieve relevant information and generate answers based on the available documents.

It was developed as part of my learning in AI and Interactive Technologies and demonstrates the practical application of Retrieval-Augmented Generation (RAG).

## Key Features

* **Semantic search:** Uses text embeddings to find relevant document passages.
* **Document-based answers:** Retrieves relevant information from the project's knowledge base.
* **Local LLM:** Uses Ollama with the Qwen3:8B model to generate answers locally.
* **Source information:** Displays the filenames of retrieved documents.
* **Performance information:** Displays vector search time.
* **Web interface:** Uses Gradio to provide a simple question-and-answer interface.

## Technologies Used

* Python
* Gradio
* Ollama
* Qwen3:8B
* Sentence Transformers
* `all-MiniLM-L6-v2`
* Vector similarity search
* Git and GitHub

## How It Works

1. The project loads the wellbeing technology documents.
2. The documents are processed into smaller text chunks.
3. Sentence Transformers converts the chunks into embeddings.
4. A user's question is converted into an embedding.
5. Semantic similarity search retrieves the most relevant text chunks.
6. The retrieved information is provided to the local LLM.
7. The application displays the generated answer, source filenames, and vector search time.

## Project Structure

```text
Wellbeing_agent/
├── main.py
├── Agent.py
├── documents/
│   └── *.txt
└── README.md
```

*The structure above is an example; adjust it to match the actual files and folders in your repository.*

## How to Run

### Requirements

* Python
* Ollama
* The Qwen3:8B model
* The Python packages required by the project

### Steps

1. Install Python and Ollama.

2. Download the model using:

   `ollama pull qwen3:8b`

3. Install the Python dependencies listed in your project's requirements file, if available.

4. Ensure the wellbeing technology text documents are in the expected folder.

5. Start the application:

   `py main.py`

6. Open the local Gradio interface in your browser.

The application may require the embedding model to be available locally before offline mode is enabled.

## Responsible Use

This project provides informational support about wellbeing and safety technologies. It is an educational prototype, not a medical diagnostic tool or a substitute for professional advice.

Answers depend on the content and quality of the available documents and should be checked against reliable sources when making important decisions.

## Current Limitations and Future Improvements

* Evaluate retrieval accuracy and answer quality.
* Improve source citations and answer grounding.
* Add automated tests and error handling.
* Measure total response time as well as vector search time.
* Explore persistent vector storage and further RAG improvements.

## Copyright

© 2026 Faiza Urooj Khan. All rights reserved.

Created for educational and portfolio purposes.
