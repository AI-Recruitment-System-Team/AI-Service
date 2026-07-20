import ollama
import json
from validator import validate_cv
def extract_resume_info(resume_text):
    prompt = f"""
You are an AI resume parser.

Extract the following information from the resume.

Return ONLY valid JSON.

Fields:
- name
- email
- phone
- skills
- education
- experience
- projects
- languages

Resume:

{resume_text}
"""

    response = ollama.chat(
        model="qwen2.5:1.5b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = response["message"]["content"]

    result = result.replace("```json", "")
    result = result.replace("```", "")
    result = result.strip()

    data = json.loads(result)

    validated_data = validate_cv(data)

    return validated_data
