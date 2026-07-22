import tempfile
import os

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

from pdf_reader import extract_text_from_pdf
from cv_parser import parse_cv_from_text
from jd_extractor import extract_jd_data
from matching_engine import match_candidate_to_job

app = FastAPI(
    title="AI Recruitment Matching Service",
    description="Independent AI service: parses resumes and job descriptions, "
                "and returns a candidate-job compatibility report.",
    version="1.0.0"
)


# =====================================================================
# Request/response models
# =====================================================================

class MatchTextRequest(BaseModel):
    resume_text: str
    job_description_text: str


# =====================================================================
# Endpoint 1: both resume and job description as plain text
# (use this when the recruiter/candidate already has typed/pasted text)
# =====================================================================

@app.post("/match")
async def match_from_text(payload: MatchTextRequest):
    """
    Input:  { "resume_text": "...", "job_description_text": "..." }
    Output: full match report (score, matched/missing skills, summary, recommendations)
    """
    cv_data = parse_cv_from_text(payload.resume_text)
    jd_data = extract_jd_data(payload.job_description_text)
    result = match_candidate_to_job(cv_data, jd_data)

    return {
        "match_result": result,
        "parsed_resume": cv_data,
        "parsed_job_description": jd_data
    }


# =====================================================================
# Endpoint 2: resume as an uploaded PDF file, job description as text
# (use this for the real-world case where a candidate uploads a CV file)
# =====================================================================

@app.post("/match-with-pdf")
async def match_from_pdf(
    resume_file: UploadFile = File(...),
    job_description_text: str = Form(...)
):
    """
    Input:  multipart form with a PDF file ("resume_file") and a text
            field ("job_description_text")
    Output: same as /match
    """
    suffix = os.path.splitext(resume_file.filename)[1] or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await resume_file.read())
        tmp_path = tmp.name

    try:
        resume_text = extract_text_from_pdf(tmp_path)
    finally:
        os.remove(tmp_path)

    cv_data = parse_cv_from_text(resume_text)
    jd_data = extract_jd_data(job_description_text)
    result = match_candidate_to_job(cv_data, jd_data)

    return {
        "match_result": result,
        "parsed_resume": cv_data,
        "parsed_job_description": jd_data
    }


# =====================================================================
# Health check - useful for the backend team to verify the service is up
# =====================================================================

@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)