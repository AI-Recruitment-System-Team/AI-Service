import ollama
import json
import re

SCHEMA_EXAMPLE = {
    "skills": ["Python", "SQL", "Docker"],
    "education": [
        {"degree": "BSc Computer Science", "institution": "Cairo University", "year": ""}
    ],
    "experience": [
        {"title": "Backend Developer", "company": "XYZ Corp", "duration": "2022-2024", "description": "Built REST APIs"}
    ],
    "projects": [
        {"name": "Chat App", "description": "Realtime chat using WebSockets"}
    ],
    "languages": ["Arabic", "English"],
    "courses": ["Machine Learning - Coursera"],
    "activities": ["Student Union Member"]
}

PROMPT_TEMPLATE = """You are a resume parsing engine. Extract structured information from the resume text below.

Return ONLY a JSON object with EXACTLY this structure (no extra text, no markdown, no explanations):

{schema}

Rules:
- education MUST always be a JSON array/list, even if there is only one entry:
  [{{"degree": "...", "institution": "...", "year": "..."}}]. NEVER return education as a single object.
- Education is not always phrased with words like "University" or "Bachelor". Lines like
  "Faculty of Artificial Intelligence - Third Year Student" or "Faculty of Engineering, Cairo University"
  ARE education entries. Put the faculty/major name in "degree" and any university name (if present) in
  "institution". Example: "Faculty of Artificial Intelligence - Third Year Student" ->
  {{"degree": "Faculty of Artificial Intelligence - Third Year Student", "institution": "", "year": ""}}
- education.year: if no graduation year is stated, use "" - do not guess or invent a year.
- If the resume has no "Experience" section but has "Projects", put projects ONLY in the projects
  field, and leave experience as an empty list [].
- List ALL skills mentioned, do not omit any, even if the skills list is long.
- courses: extract any standalone courses/certifications mentioned (e.g. "Machine Learning - Coursera").
  This is different from "education" (formal degrees) and different from "experience".
- activities: extract extracurricular activities, clubs, student unions, competitions, or volunteering
  mentioned in the resume that are NOT formal work experience.
- If a field is missing in the resume, use an empty string "" or empty list [].
- Do not invent any information not explicitly present in the resume text.

Resume text:
\"\"\"
{resume_text}
\"\"\"
"""


def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def extract_resume_data(resume_text, max_retries=3):
    resume_text = clean_text(resume_text)

    prompt = PROMPT_TEMPLATE.format(
        schema=json.dumps(SCHEMA_EXAMPLE, indent=2, ensure_ascii=False),
        resume_text=resume_text
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model="qwen2.5:1.5b",
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0, "num_predict": 1000}
            )
            result = response["message"]["content"].strip()
            data = json.loads(result)
            return _post_process(data)

        except json.JSONDecodeError as e:
            last_error = e
            print(f"[Attempt {attempt + 1}] JSON parse failed: {e}")
            continue
        except Exception as e:
            last_error = e
            print(f"[Attempt {attempt + 1}] Error: {e}")
            continue

    raise ValueError(f"Failed after {max_retries} attempts: {last_error}")


def _post_process(data):
    """Normalize field types so downstream code can always rely on the same shape."""

    education_raw = data.get("education", [])
    if isinstance(education_raw, str):
        education_raw = (
            [{"degree": education_raw, "institution": "", "year": ""}]
            if education_raw.strip() else []
        )
    if isinstance(education_raw, dict):
        education_raw = [education_raw]

    education = []
    if isinstance(education_raw, list):
        for edu in education_raw:
            if not isinstance(edu, dict):
                continue
            education.append({
                "institution": edu.get("institution") or edu.get("university") or "",
                "degree": edu.get("degree") or "",
                "year": edu.get("year") or ""
            })

    skills_raw = data.get("skills", [])
    skills = []
    for s in skills_raw:
        if isinstance(s, str):
            skills.append(s)
        elif isinstance(s, dict):
            skills.append(s.get("name", ""))
    skills = [s for s in skills if s]

    return {
        "skills": skills,
        "education": education,
        "experience": data.get("experience") or [],
        "projects": data.get("projects") or [],
        "languages": data.get("languages") or [],
        "courses": data.get("courses") or [],
        "activities": data.get("activities") or []
    }


if __name__ == "__main__":
    import fitz
    doc = fitz.open(r"E:\ai-recruitment\data\resumes\Haneen_Elabd_CV.pdf")
    text = "\n".join(page.get_text() for page in doc)
    result = extract_resume_data(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))