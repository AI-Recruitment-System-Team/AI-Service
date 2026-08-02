from pdf_reader import extract_text_from_pdf
from Contact_extractor import extract_contact, extract_stated_years_of_experience
from AI_extractor import extract_resume_data
import json


def parse_cv_from_text(cv_text):
    """
    Extracts full structured data from raw CV text (already extracted from
    a PDF, or typed/pasted directly - e.g. when used inside an API).
    """
    contact_info = extract_contact(cv_text)
    ai_data = extract_resume_data(cv_text)
    stated_years = extract_stated_years_of_experience(cv_text)

    return {
        "personal_info": contact_info,
        "skills": ai_data.get("skills", []),
        "education": ai_data.get("education", []),
        "experience": ai_data.get("experience", []),
        "projects": ai_data.get("projects", []),
        "languages": ai_data.get("languages", []),
        "courses": ai_data.get("courses", []),
        "activities": ai_data.get("activities", []),
        "stated_years_of_experience": stated_years
    }


def parse_cv(file_path):
    """
    Extracts full structured data from a single CV file (PDF).
    Thin wrapper around parse_cv_from_text - reads the PDF, then reuses
    the exact same extraction logic used everywhere else (including the API).
    """
    cv_text = extract_text_from_pdf(file_path)
    return parse_cv_from_text(cv_text)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else r"E:\ai-recruitment\data\resumes\Haneen_Elabd_CV.pdf"

    data = parse_cv(path)
    print(json.dumps(data, indent=2, ensure_ascii=False))