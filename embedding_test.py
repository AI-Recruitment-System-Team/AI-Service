import ollama
import math
import json

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2)


skill1 = "Machine Learning"
skill2 = "Cooking"
embedding1 = ollama.embeddings(
    model="nomic-embed-text",
    prompt=skill1
)["embedding"]

embedding2 = ollama.embeddings(
    model="nomic-embed-text",
    prompt=skill2
)["embedding"]

score = cosine_similarity(embedding1, embedding2)

print(score)
import ollama
import json


def extract_skills(cv_text):

    prompt = f"""
Extract technical skills from this CV.

Return ONLY JSON like this:
{{
    "skills": []
}}

CV:
{cv_text}
"""

    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response["message"]["content"]

    return json.loads(result)



skills = extract_skills(cv_text)

print(skills)