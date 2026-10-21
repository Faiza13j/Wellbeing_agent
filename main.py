import os

os.environ["HF_HUB_OFFLINE"] = "1"

import gradio as gr
import Agent
import traceback


def answer_question(question):

    try:
        if not question.strip():
            return (
                "Please enter a question.",
                "",
                ""
            )

        # ------------------------------------------
        # VECTOR SEARCH
        # ------------------------------------------

        search_start = Agent.time.time()

        results = Agent.search_documents(
            question,
            Agent.chunks,
            Agent.embeddings,
            Agent.embedding_model
        )

        search_time = Agent.time.time() - search_start

        if not results:
            return (
                "I could not find relevant information.",
                "",
                f"Vector search time: {search_time:.2f} seconds"
            )

        # ------------------------------------------
        # ASK OLLAMA
        # ------------------------------------------

        answer = Agent.ask_ollama(
            question,
            results
        )

        # ------------------------------------------
        # SOURCES
        # ------------------------------------------

        sources = "\n".join(
            f"- {result['filename']} "
            f"(similarity: {result['score']:.3f})"
            for result in results
        )

        # ------------------------------------------
        # PERFORMANCE
        # ------------------------------------------

        performance = (
            f"Vector search time: {search_time:.2f} seconds"
        )

        return (
            answer,
            sources,
            performance
        )

    except Exception as e:

        print("\nERROR:")
        traceback.print_exc()

        return (
            f"Error: {str(e)}",
            "See Command Prompt for full error.",
            f"Error occurred: {type(e).__name__}"
        )


demo = gr.Interface(
    fn=answer_question,

    inputs=gr.Textbox(
        label="Ask your question",
        placeholder="Example: What can help with medication?"
    ),

    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Sources"),
        gr.Textbox(label="Performance")
    ],

    title="Wellbeing Technology AI Agent",

    description=(
        """
### A Retrieval-Augmented Generation (RAG) AI Project

**Developed by:** Faiza Urooj Khan  
**Email:** faizaurooj.khan@edu.turkuamk.fi
© 2026 Faiza Urooj Khan. All rights reserved.

Ask questions about wellbeing and safety technologies
using information retrieved from the project's documents.

*This prototype provides informational support and is not
a medical diagnostic tool.*

"""

    )
)

demo.launch(
     server_name="127.0.0.1",
    server_port=7861,
    share=False,
    inbrowser=True
)