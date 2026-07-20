# AI Recruitment System - AI Service

## Overview
This project provides the AI backend for an AI Recruitment System.

It extracts structured information from resumes and job descriptions, then performs intelligent skill matching using semantic similarity.

## Features

- Resume Parsing
- Job Description Parsing
- Skill Extraction
- Semantic Skill Matching
- Candidate Match Score
- FastAPI REST API (Coming Soon)

## Tech Stack

- Python
- FastAPI
- Ollama
- Sentence Transformers
- Scikit-learn

## Project Structure

```
AI-Recruitment/
│
├── parser/
├── services/
├── utils/
├── data/
│
├── cv_parser.py
├── job_extractor.py
├── matcher.py
├── skill_matcher.py
├── pipeline.py
├── validator.py
├── reviewer.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python pipeline.py
```

## Status

🚧 Under Development