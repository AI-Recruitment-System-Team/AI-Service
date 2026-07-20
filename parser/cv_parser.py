from pypdf import PdfReader
import ollama
import json
import re


# =====================================
# Regex Extractors
# =====================================

def extract_name(text):
    lines = text.split("\n")

    for line in lines:
        line = line.strip()

        if len(line) > 2 and "@" not in line and "linkedin" not in line.lower():
            return line

    return ""
def extract_location(text):
    match = re.search(r"([A-Za-z ]+,\s*Egypt)", text)

    if match:
        return match.group(1)

    return ""

def extract_email(text):
    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    if match:
        return match.group()

    return ""


def extract_phone(text):
    match = re.search(
        r"(\+20|0)1[0125][0-9]{8}",
        text
    )

    if match:
        return match.group()

    return ""


def extract_linkedin(text):
    match = re.search(
        r"https?://(?:[a-z]{2}\.)?linkedin\.com/\S+",
        text
    )

    if match:
        return match.group()

    return ""


def extract_github(text):
    match = re.search(
        r"https?://(?:www\.)?github\.com/\S+",
        text
    )

    if match:
        return match.group()

    return ""


def extract_portfolio(text):
    match = re.search(
        r"https?://\S+",
        text
    )

    if match:
        url = match.group()

        if "linkedin" not in url and "github" not in url:
            return url

    return ""


# =====================================
# PDF Reader
# =====================================

def extract_text_from_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =====================================
# AI Resume Parser
# =====================================

def parse_cv(cv_text):

    print("Sending CV to AI...\n")

    prompt = """
You are an expert resume information extraction model.

Read the entire resume carefully before answering.

Rules:
- Return ONLY valid JSON.
- No explanations.
- No markdown.
- Do not invent information.
- Extract information exactly from the resume.
- If a section exists in the resume, you MUST extract it.
- Never return empty skills or projects when they appear in the resume.
For each skill:
- Extract the skill name exactly.
- Assign a general category based on the skill meaning.
- Categories should be generated dynamically.
- Do not use fixed categories.
- Examples: Programming Language, Machine Learning, Database, Cloud, Framework, Tool, Soft Skill.
Important:
- Skills must be extracted from Technical Skills / Skills sections.
- Projects must be extracted from Projects section.
- Do not confuse section titles with values.
- "Third Year Student" is education status, not a degree.
- Do not use "Technical Skills" as field_of_study.
- Keep project title and description separate.

{{{
 "skills": [],

 "education":[
   {
    "institution":"",
    "degree":"",
    "field_of_study":"",
    "start_year":"",
    "end_year":"",
    "details":""
   }
 ],

 "experience":[
   {
    "type":"",
    "title":"",
    "organization":"",
    "start_date":"",
    "end_date":"",
    "description":""
   }
 ],

 "projects":[
   {
    "title":"",
    "description":"",
    "technologies":[]
   }
 ],

 "certifications":[
   {
    "name":"",
    "issuer":"",
    "date":""
   }
 ],

 "languages":[
   {
    "language":"",
    "level":""
   }
 ],

 "summary":"",
 "achievements":[],
 "volunteering":[],
 "interests":[]
}

Resume:

RESUME_TEXT_HERE
"""
    prompt = prompt.replace("RESUME_TEXT_HERE", cv_text)

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

    result = re.sub(r"//.*", "", result)

    print("========== RAW RESPONSE ==========")
    print(result)
        # =====================================
    # Extract JSON
    # =====================================

    match = re.search(r"\{[\s\S]*\}", result)

    if not match:
        raise Exception("No JSON object found in model response.")

    json_text = match.group(0)

    try:
        ai_data = json.loads(json_text)
        # Remove empty experience items
        ai_data["experience"] = [
        exp for exp in ai_data.get("experience", [])
        if any(exp.values())]


# Remove empty certifications
        ai_data["certifications"] = [
        cert for cert in ai_data.get("certifications", [])
        if any(cert.values())
        ]


# Normalize skills
        for skill in ai_data.get("skills", []):

         if "level" in skill:
            skill["confidence"] = 5
            skill.pop("level")

         if "category" not in skill:
            skill["category"] = ""

    except json.JSONDecodeError:

        print("\n========== INVALID JSON ==========\n")
        print(json_text)

        raise Exception("The AI returned invalid JSON.")

    # =====================================
    # Merge Regex + AI Results
    # =====================================

        # =====================================
    # Merge Regex + AI Results
    # =====================================

    final_result = {

        "personal_info": {

            "name": extract_name(cv_text),

            "email": extract_email(cv_text),

            "phone": extract_phone(cv_text),

            "location": extract_location(cv_text),

            "links": {

                "linkedin": extract_linkedin(cv_text),

                "github": extract_github(cv_text),

                "portfolio": extract_portfolio(cv_text)

            }
        },


        "summary": ai_data.get("summary", ""),


        "skills": ai_data.get("skills", []),


        "education": ai_data.get("education", []),


        "experience": ai_data.get("experience", []),


        "projects": ai_data.get("projects", []),


        "certifications": ai_data.get("certifications", []),


        "languages": ai_data.get("languages", []),


        "achievements": ai_data.get("achievements", []),


        "volunteering": ai_data.get("volunteering", []),


        "interests": ai_data.get("interests", [])

    }


    return final_result


# =====================================
# Main
# =====================================

if __name__ == "__main__":

    file_path = r"E:\ai-recruitment\data\resumes\Haneen_Elabd_CV.pdf"

    print("Reading PDF...\n")

    cv_text = extract_text_from_pdf(file_path)

    print("========== CV TEXT ==========\n")
    print(cv_text)

    print("\n========== PARSED CV ==========\n")

    cv_data = parse_cv(cv_text)

    print("\n========== FINAL RESULT ==========\n")


    print(json.dumps(cv_data, indent=4, ensure_ascii=False))