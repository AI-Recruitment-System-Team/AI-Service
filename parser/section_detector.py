import ollama
import json
import re


def detect_sections_with_ai(text):

    prompt = f"""
You are an expert Resume Analyzer.

Your task is to detect all sections in this resume.

Rules:
- Do NOT use predefined section names.
- Detect all sections that exist in the resume.
- Keep the original content under each section.
- Do not summarize.
- Do not remove information.
- Return ONLY valid JSON.
- Do not add explanations.

Resume:

{text}

Return format:

{{
    "Section Name": "content"
}}
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

    if match:
        return json.loads(match.group(0))

    return {}



def extract_header(text, sections):

    """
    Extract the information before the first detected section.
    Usually contains:
    - Name
    - Email
    - Phone
    - Location
    - Links
    """

    if not sections:
        return {
            "header": text,
            "sections": {}
        }


    first_section_position = len(text)


    first_section_name = None


    for section_name in sections.keys():

        position = text.lower().find(section_name.lower())

        if position != -1 and position < first_section_position:

            first_section_position = position
            first_section_name = section_name



    header_text = text[:first_section_position].strip()


    return {

        "header": header_text,

        "sections": sections

    }



def detect_resume_structure(text):

    sections = detect_sections_with_ai(text)

    result = extract_header(text, sections)

    return result



# -----------------------
# Test
# -----------------------

from pypdf import PdfReader


file_path = r"E:\ai-recruitment\data\resumes\Haneen_Elabd_CV.pdf"


reader = PdfReader(file_path)


text = ""


for page in reader.pages:

    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"



resume = detect_resume_structure(text)



print("=" * 40)
print("HEADER")
print(resume["header"])



for name, content in resume["sections"].items():

    print("=" * 40)
    print(name)

    print(content)