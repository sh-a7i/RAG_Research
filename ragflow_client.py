
import requests
import time


def ask_ragflow(
    question: str,
    chat_id: str,
    api_key: str,
    base_url: str
):
    """
    Send a single-turn question to a RAGFlow Chat Assistant.

    The assistant's saved retrieval configuration determines
    keyword weight, similarity threshold, Top N, etc.
    """

    url = (
        f"{base_url.rstrip('/')}/api/v1/openai/"
        f"{chat_id}/chat/completions"
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "model",
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ],
        "stream": False,
        "extra_body": {
            "reference": True
        }
    }

    start = time.perf_counter()

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=180
    )

    latency = time.perf_counter() - start

    # Raise HTTP errors (401, 404, 500, etc.)
    response.raise_for_status()

    result = response.json()

    # OpenAI-compatible response format
    choices = result.get("choices", [])

    if not choices:
        raise RuntimeError(
            f"No answer returned by RAGFlow: {result}"
        )

    message = choices[0].get("message", {})

    answer = message.get("content", "")

    if not answer:
        raise RuntimeError(
            f"Empty answer returned by RAGFlow: {result}"
        )

    references = message.get("reference", {})

    return {
        "answer": answer,
        "references": references,
        "latency_seconds": round(latency, 4),
        "usage": result.get("usage", {}),
        "raw_response": result
    }
