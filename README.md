# AI Recruitment Matching Service

> **Status: MVP**

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
  - an AI-generated candidate summary (strengths-only from the model;
    gaps are always computed deterministically, never invented by the AI)
  - recommendations for the recruiter

## Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com)** installed and running locally, with the
   `qwen2.5:3b` model pulled:
   ```bash
   ollama pull qwen2.5:3b
   ```
3. Python packages (see `requirements.txt`)

## Setup

```bash
# create and activate a virtual environment
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
# source venv/bin/activate     # macOS/Linux

# install dependencies
pip install -r requirements.txt
```

First run will also download the `all-MiniLM-L6-v2` embedding model
(used for semantic matching and section detection) and, if `gliner` is
installed, a GLiNER model - both require internet access once, then work
offline from a local cache.

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
  "match_score": 71.8,
  "matched_skills": ["Python", "Google Ads", "..."],
  "missing_skills": ["Docker"],
  "candidate_summary": "...",
  "recommendations": ["..."]
}
```

## Project structure

Everything lives in one flat folder so every file can simply
`import` from any other, with no path configuration needed:

```
ai_service/
├── main.py                  # FastAPI app - the entry point / API
├── pdf_reader.py             # PDF -> text extraction (PyMuPDF)
├── contact_extractor.py      # regex-based contact info + stated years-of-experience
├── ai_extractor.py           # AI-based CV field extraction (skills, education, experience...)
├── cv_parser.py               # combines the above into one structured CV
├── batch_processor.py         # CLI tool: parse every resume in a folder
├── jd_extractor.py            # AI-based job description extraction
├── skills_matcher.py          # keyword-based skill matcher (safety net for the AI)
├── skills_taxonomy.json       # editable list of known skills/aliases (~225 entries)
├── matching_engine.py          # scores a CV against a JD (skills/experience/education/semantic)
├── weights_config.json         # editable scoring weights
└── requirements.txt
```

## How extraction works (three layers, not one)

Relying on a single small local LLM (`qwen2.5:3b`) for everything proved
unreliable on its own, so extraction is layered:

1. **Fixed taxonomy** (`skills_matcher.py` + `skills_taxonomy.json`) —
   100% reliable for any skill already in the list, regardless of where
   it appears in the text (not limited to a "Skills:" section).
2. **The LLM** (`qwen2.5:3b`) — handles everything that needs contextual
   understanding: which lines are jobs vs. projects, what counts as an
   activity vs. a course, etc.
3. **GLiNER** *(optional)* — a specialized zero-shot entity-extraction
   model, used as a third pass over skills specifically. If it isn't
   installed, this layer is silently skipped and the other two still work.

Section boundaries (where "Experience" ends and "Projects" begins, etc.)
are found the same layered way: an exact-match list of common header
phrasings first (fast), falling back to embedding-based *semantic*
matching for header phrasing never seen before (e.g. "My Journey"
instead of "Experience") - so a novel resume format doesn't silently
break extraction.

## Configuration

- **`skills_taxonomy.json`** — add a new skill by adding one line, no code
  changes needed. Restart the service for changes to take effect.
- **`weights_config.json`** — adjust how much each factor (skills,
  experience, education, semantic fit) contributes to the final match score.

## Known Limitations

This is a local-LLM-based system, not a guarantee of perfect extraction.
Tested across multiple domains (tech, marketing, data analytics) and
resume formats, with these known gaps:

- **Table-formatted resumes** are not specifically handled — text
  extracted from PDF tables can come out jumbled, since `pdf_reader.py`
  extracts plain text without layout/column awareness.
- **Mixed-language resumes** (Arabic/English interleaved in the same
  sentence) haven't been tested and may reduce extraction accuracy.
- **GPA is not currently extracted** - no field for it exists in the schema.
- The `skills_taxonomy.json` list, while broad, is not exhaustive — a
  brand-new or highly niche skill name may only be caught by the LLM or
  GLiNER layers, not guaranteed by the fixed list.
- Years-of-experience is calculated from actual job dates when available
  (falls back to an explicitly stated "X years of experience" phrase only
  if no dates could be parsed at all) - a resume with neither will score
  0 years, which may unfairly affect candidates whose experience just
  wasn't in a recognizable date format.
- This service should be treated as **a starting point for a recruiter
  to review, not a final automated decision** - a human-review step
  before any hiring decision is strongly recommended.


  results.
- Requires Ollama running locally (or reachable) with `qwen2.5:3b`
  available, plus internet access on first run to download the
  embedding model(s) (cached locally afterward).
- Each `/match` call takes roughly 15–30 seconds due to multiple model
  calls involved - not suitable for a synchronous request the user waits
  on live; consider a background job / polling pattern for production use.
