# AI Recruitment Matching Service

> **🚧 Status: Under Development** 
An independent AI service that parses resumes and job descriptions, and
compares them to produce a compatibility report for recruiters.

## What it does

- **Resume parsing** — extracts skills, education, experience, projects,
  languages, courses, and activities from a CV (PDF or plain text).
- **Job description analysis** — extracts required/preferred skills,
  education requirements, experience needed, and responsibilities.
- **Candidate-job matching** — compares a parsed resume against a parsed
  job description and returns:
  - an overall match score (0–100%) with a breakdown by category
  - matched skills / missing skills
  - an AI-generated candidate summary
  - recommendations for the recruiter

## Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com)** installed and running locally, with the
   `qwen2.5:1.5b` model pulled:
   ```bash
   ollama pull qwen2.5:1.5b
   ```
3. Python packages (see below)

## Setup

```bash
# create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate     # macOS/Linux

# install dependencies
pip install -r requirements.txt
```

## Running the service

```bash
python main.py
```

or, for auto-reload during development:

```bash
uvicorn main:app --reload
```

The service starts at `http://localhost:8000`.
Interactive API docs (try it directly from the browser): `http://localhost:8000/docs`

## API Endpoints

### `POST /match`
Match a resume and job description given as plain text.

**Request body:**
```json
{
  "resume_text": "full text of the candidate's CV...",
  "job_description_text": "full text of the job posting..."
}
```

### `POST /match-with-pdf`
Same as above, but the resume is uploaded as a PDF file
(multipart form: `resume_file` = PDF file, `job_description_text` = text field).

### `GET /health`
Simple health check, returns `{"status": "ok"}`.

### Example response (both endpoints)
```json
{
  "match_result": {
    "match_score": 83.6,
    "score_breakdown": {
      "skills": 87.5,
      "experience": 100.0,
      "education": 83.4,
      "semantic_overall": 49.2
    },
    "matched_skills": ["Python", "Machine Learning", "..."],
    "missing_skills": ["Docker"],
    "candidate_years_of_experience": 2,
    "required_years_of_experience": 2,
    "candidate_summary": "...",
    "recommendations": ["..."]
  },
  "parsed_resume": { "...": "full structured CV data" },
  "parsed_job_description": { "...": "full structured JD data" }
}
```

## Project structure

```
ai_service/
├── main.py                 # FastAPI app - the entry point / API
├── pdf_reader.py            # PDF -> text extraction (PyMuPDF)
├── contact_extractor.py     # regex-based contact info extraction (CV)
├── ai_extractor.py          # AI-based CV field extraction (skills, education...)
├── cv_parser.py             # combines the above into one structured CV
├── batch_processor.py       # CLI tool: parse every resume in a folder
├── jd_extractor.py          # AI-based job description extraction
├── skills_matcher.py        # keyword-based skill matcher (safety net for the AI)
├── skills_taxonomy.json     # editable list of known skills/aliases
├── matching_engine.py       # scores a CV against a JD (skills/experience/education/semantic)
├── weights_config.json      # editable scoring weights
└── requirements.txt
```

## Configuration

- **`skills_taxonomy.json`** — add a new skill by adding one line, no code
  changes needed. Restart the service for changes to take effect.
- **`weights_config.json`** — adjust how much each factor (skills,
  experience, education, semantic fit) contributes to the final match score.

## Notes for the backend team

- This service is stateless — it doesn't store any data. The backend is
  responsible for persisting parsed resumes, job postings, and match
  results.
- Requires Ollama running locally (or reachable) with `qwen2.5:1.5b`
  available, plus internet access on first run to download the
  `sentence-transformers` embedding model (cached locally afterward).