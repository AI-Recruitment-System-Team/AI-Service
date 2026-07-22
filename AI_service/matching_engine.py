import json
import os
import re
from skills_matcher import extract_all_skills

# =====================================================================
# CONFIG LOADING
# =====================================================================

_DEFAULT_WEIGHTS = {
    "skills": 0.5,
    "experience": 0.2,
    "education": 0.15,
    "semantic_overall": 0.15
}

_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights_config.json")


def _load_default_weights():
    try:
        with open(_WEIGHTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # if the config file is missing or broken, fall back to safe defaults
        # rather than crashing the whole matching process
        return dict(_DEFAULT_WEIGHTS)


# =====================================================================
# EMBEDDING MODEL (loaded once, lazily, only when actually needed)
# =====================================================================

_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _cosine_similarity(text_a, text_b):
    """Returns a 0-1 similarity score between two pieces of text."""
    if not text_a.strip() or not text_b.strip():
        return 0.0

    from sentence_transformers import util
    model = _get_embedding_model()
    emb_a = model.encode(text_a, convert_to_tensor=True)
    emb_b = model.encode(text_b, convert_to_tensor=True)
    score = util.cos_sim(emb_a, emb_b).item()
    return max(0.0, min(1.0, score))


# =====================================================================
# 1. SKILLS SCORE
# =====================================================================

def _score_skills(cv_data, jd_data, semantic_threshold=0.6):
    candidate_raw_skills = cv_data.get("skills", [])
    candidate_text_blob = " ".join(candidate_raw_skills)
    candidate_canonical = set(extract_all_skills(candidate_text_blob))

    required = set(jd_data.get("required_skills", []))
    nice = set(jd_data.get("nice_to_have_skills", []))

    matched_required = candidate_canonical & required
    matched_nice = candidate_canonical & nice
    missing_required = required - candidate_canonical

    # semantic fallback: for required skills not found by exact/canonical
    # matching, check if any of the candidate's raw skill phrases are
    # close enough in meaning (catches synonyms outside the fixed taxonomy)
    semantic_matched = set()
    still_missing = set()

    for skill in missing_required:
        best_score = 0.0
        for candidate_skill in candidate_raw_skills:
            best_score = max(best_score, _cosine_similarity(skill, candidate_skill))
        if best_score >= semantic_threshold:
            semantic_matched.add(skill)
        else:
            still_missing.add(skill)

    total_weight = len(required) * 1.0 + len(nice) * 0.5
    achieved_weight = (
        len(matched_required) * 1.0
        + len(semantic_matched) * 0.7
        + len(matched_nice) * 0.5
    )

    score = (achieved_weight / total_weight) if total_weight > 0 else 1.0
    score = max(0.0, min(1.0, score))

    matched_skills = sorted(matched_required | semantic_matched | matched_nice)
    missing_skills = sorted(still_missing)

    return score, matched_skills, missing_skills


# =====================================================================
# 2. EXPERIENCE SCORE
# =====================================================================

def _parse_required_years(years_text):
    """Extracts the minimum required years as a number from strings like
    "2+ years", "2-4 years", "3 years", "entry-level", "" """
    if not years_text:
        return 0

    match = re.search(r"(\d+)", years_text)
    if match:
        return int(match.group(1))

    return 0  # entry-level / unspecified -> no numeric barrier


def _parse_candidate_years(experience_entries):
    """Sums up years from duration strings like "2022-2024" or "2020-Present"."""
    import datetime
    current_year = datetime.datetime.now().year
    total_years = 0.0

    for entry in experience_entries:
        if not isinstance(entry, dict):
            continue
        duration = entry.get("duration", "") or ""
        years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", duration)]

        if "present" in duration.lower() and years:
            total_years += max(0, current_year - years[0])
        elif len(years) >= 2:
            total_years += max(0, years[-1] - years[0])
        elif len(years) == 1:
            total_years += 1  # a single year mentioned - assume ~1 year role

    return total_years


def _score_experience(cv_data, jd_data):
    required_years = _parse_required_years(jd_data.get("years_of_experience", ""))
    candidate_years = _parse_candidate_years(cv_data.get("experience", []))

    if required_years == 0:
        return 1.0, candidate_years, required_years

    score = min(1.0, candidate_years / required_years)
    return score, candidate_years, required_years


# =====================================================================
# 3. EDUCATION SCORE
# =====================================================================

def _score_education(cv_data, jd_data):
    requirements = jd_data.get("education_requirements", [])
    if not requirements:
        return 1.0  # job doesn't require a specific education background

    candidate_education = cv_data.get("education", [])
    if not candidate_education:
        return 0.0  # job requires education, candidate listed none

    candidate_text = " ".join(
        f"{e.get('degree', '')} {e.get('institution', '')}".strip()
        for e in candidate_education if isinstance(e, dict)
    )
    requirement_text = " ".join(requirements)

    return _cosine_similarity(candidate_text, requirement_text)


# =====================================================================
# 4. OVERALL SEMANTIC FIT
# =====================================================================

def _build_cv_text(cv_data):
    parts = []
    parts.extend(cv_data.get("skills", []))
    for exp in cv_data.get("experience", []):
        if isinstance(exp, dict):
            parts.append(exp.get("description", ""))
    for proj in cv_data.get("projects", []):
        if isinstance(proj, dict):
            parts.append(f"{proj.get('name', '')} {proj.get('description', '')}")
    parts.extend(cv_data.get("courses", []))
    for edu in cv_data.get("education", []):
        if isinstance(edu, dict):
            parts.append(edu.get("degree", ""))
    return " ".join(p for p in parts if p)


def _build_jd_text(jd_data):
    parts = [jd_data.get("job_title", "")]
    parts.extend(jd_data.get("responsibilities", []))
    parts.extend(jd_data.get("required_skills", []))
    parts.extend(jd_data.get("nice_to_have_skills", []))
    return " ".join(p for p in parts if p)


def _score_semantic_overall(cv_data, jd_data):
    cv_text = _build_cv_text(cv_data)
    jd_text = _build_jd_text(jd_data)
    return _cosine_similarity(cv_text, jd_text)


# =====================================================================
# RECOMMENDATIONS (rule-based, no AI call needed)
# =====================================================================

def _build_recommendations(missing_skills, candidate_years, required_years, education_score):
    recommendations = []

    if missing_skills:
        skills_list = ", ".join(missing_skills)
        recommendations.append(
            f"Candidate is missing these required skills: {skills_list}. "
            f"Consider asking about related experience during the interview."
        )

    if required_years > 0 and candidate_years < required_years:
        recommendations.append(
            f"Candidate has ~{candidate_years:.0f} year(s) of experience vs "
            f"{required_years} required. May need additional ramp-up time."
        )

    if education_score < 0.4:
        recommendations.append(
            "Candidate's educational background doesn't closely match the "
            "stated requirement - worth verifying relevance during screening."
        )

    if not recommendations:
        recommendations.append("Candidate appears to be a strong match on paper.")

    return recommendations


# =====================================================================
# AI-GENERATED CANDIDATE SUMMARY
# =====================================================================

def generate_candidate_summary(cv_data, jd_data, matched_skills, missing_skills):
    """
    Generates a short candidate summary for the recruiter. To avoid the
    small model inventing gaps that don't exist, the AI is only ever asked
    to describe STRENGTHS (a positive-only task it handles reliably).
    The weakness/gap sentence is built deterministically from our own
    already-computed missing_skills list - never generated by the model.
    """
    strengths_sentence = _generate_strengths_sentence(cv_data, jd_data, matched_skills)

    if missing_skills:
        weakness_sentence = (
            f"However, the profile does not show experience with: "
            f"{', '.join(missing_skills)}."
        )
    else:
        weakness_sentence = (
            "No gaps were identified against this role's required skills."
        )

    return f"{strengths_sentence} {weakness_sentence}"


def _generate_strengths_sentence(cv_data, jd_data, matched_skills):
    """Asks the model for a purely positive 1-2 sentence description of the
    candidate's relevant strengths. Never asked about weaknesses, so there's
    nothing negative for it to hallucinate."""
    try:
        import ollama

        cv_summary_input = {
            "skills": cv_data.get("skills", []),
            "education": cv_data.get("education", []),
            "experience": cv_data.get("experience", []),
            "projects": cv_data.get("projects", [])
        }

        prompt = f"""You are helping a recruiter understand a candidate's strengths for a role.

Job title: {jd_data.get("job_title", "")}
Candidate profile:
{json.dumps(cv_summary_input, ensure_ascii=False)}

Skills the candidate has that are relevant to this job: {", ".join(matched_skills) if matched_skills else "none identified"}

Write ONLY 1-2 sentences describing the candidate's strengths and relevant background
for this specific role, based strictly on the profile above. Do not mention anything
the candidate lacks or is missing - only describe what they have. Do not invent any
skill, project, or experience not shown above. Return ONLY the sentence(s), no
headings, no markdown, no extra commentary.
"""

        response = ollama.chat(
            model="qwen2.5:1.5b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 150}
        )

        summary = response["message"]["content"].strip()
        return summary if summary else _build_fallback_strengths(cv_data, jd_data, matched_skills)

    except Exception:
        return _build_fallback_strengths(cv_data, jd_data, matched_skills)


def _build_fallback_strengths(cv_data, jd_data, matched_skills):
    name = cv_data.get("personal_info", {}).get("name", "The candidate")
    job_title = jd_data.get("job_title", "this role")
    if matched_skills:
        return f"{name} shows relevant strengths in {', '.join(matched_skills)} for the {job_title} position."
    return f"{name} has a general profile submitted for the {job_title} position."


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def match_candidate_to_job(cv_data, jd_data, weights=None):
    """
    Compares a parsed CV (from cv_parser.py) against a parsed JD
    (from jd_extractor.py) and returns a compatibility report.
    """
    weights = weights or _load_default_weights()

    skills_score, matched_skills, missing_skills = _score_skills(cv_data, jd_data)
    experience_score, candidate_years, required_years = _score_experience(cv_data, jd_data)
    education_score = _score_education(cv_data, jd_data)
    semantic_score = _score_semantic_overall(cv_data, jd_data)

    final_score = (
        skills_score * weights.get("skills", 0)
        + experience_score * weights.get("experience", 0)
        + education_score * weights.get("education", 0)
        + semantic_score * weights.get("semantic_overall", 0)
    )

    recommendations = _build_recommendations(
        missing_skills, candidate_years, required_years, education_score
    )

    candidate_summary = generate_candidate_summary(
        cv_data, jd_data, matched_skills, missing_skills
    )

    return {
        "match_score": round(final_score * 100, 1),
        "score_breakdown": {
            "skills": round(skills_score * 100, 1),
            "experience": round(experience_score * 100, 1),
            "education": round(education_score * 100, 1),
            "semantic_overall": round(semantic_score * 100, 1)
        },
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "candidate_years_of_experience": round(candidate_years, 1),
        "required_years_of_experience": required_years,
        "candidate_summary": candidate_summary,
        "recommendations": recommendations
    }


if __name__ == "__main__":
    sample_cv = {
        "skills": ["Python", "C++", "Machine Learning basics", "Data Analysis",
                    "Computer Vision basics", "Web Development"],
        "education": [{"degree": "Faculty of Artificial Intelligence - Third Year Student",
                        "institution": "", "year": ""}],
        "experience": [],
        "projects": [
            {"name": "Clinic Booking Website",
             "description": "Web application for booking doctor appointments"},
            {"name": "Calorie Calculator ML Project",
             "description": "Machine learning model for calorie estimation"}
        ],
        "courses": ["Machine Learning - Coursera"]
    }

    sample_jd = {
        "job_title": "Machine Learning Intern",
        "required_skills": ["Python", "Machine Learning", "Data Analysis"],
        "nice_to_have_skills": ["Computer Vision", "Docker"],
        "education_requirements": ["Faculty of Artificial Intelligence or Computer Science"],
        "years_of_experience": "",
        "responsibilities": ["Assist in building and training ML models"]
    }

    result = match_candidate_to_job(sample_cv, sample_jd)
    print(json.dumps(result, indent=2, ensure_ascii=False))