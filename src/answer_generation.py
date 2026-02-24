import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi3"

def generate_answer(context_chunks, question,max_tokens):
    context = "\n\n".join(context_chunks)

    prompt = f"""
    You are a document-based question answering system.

    Rules:
    - Use ONLY the information explicitly stated in the context.
    - Be extremely CONCISE and DIRECT.
    - Do NOT infer, explain, or rephrase.
    - If the question cannot be answered using exact facts from the context,
    reply exactly:
    "Answer not found in the document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    print("Generating answer...", flush=True)
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens
            },
            "keep_alive": "10m"
        },
        stream=True
    )

    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if "response" in data:
                yield data["response"]
            if data.get("done"):
                break