import ollama
import json
import re


def extract_resume_data(sections):

    resume_text = ""

    for section, content in sections.items():
        resume_text += f"\n{section}:\n{content}\n"


    prompt = f"""
You are an AI Resume Parser for a recruitment system.

Your task is to extract structured information from ANY resume.

The resume can belong to:
- Student
- Fresh graduate
- Experienced employee
- Freelancer
- Researcher
- Any profession or industry

Rules:
1. Return ONLY valid JSON.
2. Do not add explanations.
3. Do not use markdown.
4. Do not guess information.
5. Extract only information explicitly mentioned.
6. If information is missing return "" or [].
7. Keep original wording from the resume.
8. Do not create skills, companies, projects, or technologies that are not mentioned.
9. Separate project title from description.
10. Extract technologies only if they are explicitly written.

Skills classification rules:

- Assign an accurate category for each skill.
- Categories must describe the skill domain.

Use categories like:
- Programming Language
- Artificial Intelligence
- Data Science
- Database
- Framework or Library
- Cloud Platform
- Tool
- Soft Skill

Examples:
Python -> Programming Language
C++ -> Programming Language
Machine Learning -> Artificial Intelligence
Deep Learning -> Artificial Intelligence
Computer Vision -> Artificial Intelligence
Data Analysis -> Data Science
SQL -> Database
TensorFlow -> Framework or Library
Web Development-> Software Development
Important:
- Do NOT classify technical skills as Soft Skills.
- Soft Skills are only communication, teamwork, leadership, problem solving, etc.
- Extract all skills mentioned in the resume.
- Confidence score:
    10 = explicitly listed in skills section
    8 = mentioned in experience/projects
    5 = indirectly mentioned
    0 = uncertain

Experience rules:
- Include jobs, internships, freelance, research, and volunteering.
- Do not assume a student activity is employment.

Education rules:
- Do not confuse degree with student status.
- Extract institution, degree, field, and dates when available.


Return exactly this JSON structure:

{{
  "personal_info": {{
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
    "links": {{
      "linkedin": "",
      "github": "",
      "portfolio": "",
      "other": []
    }}
  }},

  "summary": "",

  "skills": [
    {{
      "name": "",
      "category": "",
      "confidence": 0
    }}
  ],

  "education": [
    {{
      "institution": "",
      "degree": "",
      "field_of_study": "",
      "start_year": "",
      "end_year": "",
      "details": ""
    }}
  ],

  "experience": [
    {{
      "type": "",
      "title": "",
      "organization": "",
      "start_date": "",
      "end_date": "",
      "description": ""
    }}
  ],

  "projects": [
    {{
      "title": "",
      "description": "",
      "technologies": []
    }}
  ],

  "certifications": [
    {{
      "name": "",
      "issuer": "",
      "date": ""
    }}
  ],

  "languages": [
    {{
      "language": "",
      "level": ""
    }}
  ],

  "achievements": [],

  "volunteering": [],

  "interests": []
}}


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
        ],
        options={
            "temperature": 0
        }
    )


    result = response["message"]["content"]

    result = result.replace("```json", "")
    result = result.replace("```", "").strip()


    match = re.search(r"\{[\s\S]*\}", result)


    if not match:
        print(result)
        return None


    json_text = match.group(0)


    try:
        return json.loads(json_text)

    except json.JSONDecodeError as e:

        print("JSON ERROR:")
        print(e)
        print(json_text)

        return None