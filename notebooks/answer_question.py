
from google.colab import userdata
from google import genai

import runpy

PROJECT_PATH = "/content/drive/MyDrive/Agentic_Deep_Research"

retriever = runpy.run_path(
    f"{PROJECT_PATH}/notebooks/reranked_retriever.py"
)

retrieve = retriever["retrieve"]

api_key = userdata.get(
    "GeminiAPIKey"
)

client = genai.Client(
    api_key=api_key
)

def answer_question(question):

    hits = retrieve(
        question,
        final_k=5
    )

    evidence = ""

    for i, hit in enumerate(hits):

        evidence += f"""

[Evidence {i+1}]
Paper: {hit["arxiv_id"]}

{hit["text"][:2000]}

"""

    prompt = f"""
Answer the question using ONLY the evidence.

QUESTION:
{question}

EVIDENCE:
{evidence}

Requirements:
- Be factually grounded.
- If the answer is not in evidence, say so.
- Maximum 150 words.
- Academic style.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text
