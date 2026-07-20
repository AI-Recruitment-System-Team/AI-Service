import ollama
import json
import re


# =====================================
# Job Description Parser
# =====================================

def parse_job(job_text):

    print("Sending Job Description to AI...\n")

    prompt = f"""
You are an AI Job Description Parser.

Extract structured information from the job description.

Rules:
- Return ONLY valid JSON.
- No explanations.
- No markdown.
- Do not guess missing information.
- Extract all skills mentioned.
- Assign an accurate technical category for each skill.
- Categories must describe the skill domain, not just "Technical".
- Do not classify technical skills as soft skills.

Use categories like:
- Programming Language
- Artificial Intelligence
- Machine Learning
- Data Science
- Database
- Framework
- Library
- Cloud Platform
- Tool
- Soft Skill
Use these rules:

- Programming languages:
  Python, C++, Java, JavaScript -> Programming Language

- AI/ML:
  Machine Learning, Deep Learning, Computer Vision, NLP -> Artificial Intelligence

- Data:
  Data Analysis, Data Science, Pandas, NumPy -> Data Science

- Databases:
  SQL, MongoDB, PostgreSQL -> Database

- Frameworks/Libraries:
  TensorFlow, PyTorch, React, Django -> Framework or Library

- Soft Skills ONLY include:
  Communication, Leadership, Teamwork, Problem Solving

Never classify technical skills like Data Analysis, SQL, Machine Learning as Soft Skills.
Examples:
Python -> Programming Language
C++ -> Programming Language
Machine Learning -> Artificial Intelligence
Deep Learning -> Artificial Intelligence
TensorFlow -> Framework
SQL -> Database
Communication -> Soft Skill

Return this exact JSON schema:

{{
    "job_title": "",
    "company": "",
    "required_skills": [
        {{
            "name": "",
            "category": ""
        }}
    ],
    "preferred_skills": [
        {{
            "name": "",
            "category": ""
        }}
    ],
    "experience_required": "",
    "education_required": "",
    "responsibilities": [],
    "description": ""
}}

Job Description:

{job_text}
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

    print("========== RAW RESPONSE ==========")
    print(result)


    match = re.search(r"\{[\s\S]*\}", result)

    if not match:
        raise Exception("No JSON found")

    json_text = match.group(0)


    return json.loads(json_text)



# =====================================
# Test
# =====================================

if __name__ == "__main__":

    job_description = """
    Machine Learning Engineer Intern

    Requirements:
    - Python
    - Machine Learning
    - Data Analysis
    - TensorFlow
    - SQL
    - Basic knowledge of Deep Learning

    Responsibilities:
    - Build machine learning models
    - Clean and analyze datasets
    - Work with AI team

    Education:
    Computer Science or Artificial Intelligence student
    """

    job_data = parse_job(job_description)

    print("\n========== FINAL JOB ==========\n")

    print(json.dumps(
        job_data,
        indent=4,
        ensure_ascii=False
    ))