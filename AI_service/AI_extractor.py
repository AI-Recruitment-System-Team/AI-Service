import ollama
import json
import re
from skills_matcher import extract_all_skills

# =====================================================================
# GLiNER: a specialized zero-shot Named Entity Recognition model, used
# here as a THIRD layer of skill extraction alongside the fixed taxonomy
# (skills_matcher.py) and the LLM (qwen). Unlike qwen, GLiNER isn't a
# chat model being asked to behave like an extractor - it's purpose-built
# for entity extraction, so it can catch skills that are neither in our
# taxonomy nor caught by the model's own reasoning.
#
# Setup required: pip install gliner
# If GLiNER isn't installed or fails to load, this degrades gracefully -
# the rest of the extraction pipeline (qwen + skills_matcher) keeps
# working exactly as before, unaffected.
# =====================================================================

_gliner_model = None


def _get_gliner_model():
    global _gliner_model
    if _gliner_model is None:
        from gliner import GLiNER
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    return _gliner_model


def extract_skills_with_gliner(text):
    """
    Uses GLiNER to pull out skill/technology/tool mentions from raw resume
    text. Returns a list of strings (not yet canonicalized - that happens
    later when merged with the taxonomy). Returns [] if GLiNER isn't
    installed or errors out, so callers never need to handle failure.
    """
    try:
        model = _get_gliner_model()
        labels = ["skill", "technology", "programming language", "software tool", "framework"]
        entities = model.predict_entities(text, labels, threshold=0.4)
        return [e["text"] for e in entities if e.get("text")]
    except ImportError:
        # GLiNER not installed - silently skip, this layer is optional
        return []
    except Exception as e:
        print(f"GLiNER extraction skipped due to an error: {e}")
        return []

SCHEMA_EXAMPLE = {
    "skills": ["<skill from text>", "..."],
    "education": [
        {"degree": "<degree from text>", "institution": "<institution from text or empty>", "year": "<year from text or empty>"}
    ],
    "projects": [
        {"name": "<project name from text>", "description": "<project description from text>"}
    ],
    "languages": ["<language from text>", "..."],
    "courses": ["<course from text>", "..."],
    "activities": ["<activity from text>", "..."]
}

PROMPT_TEMPLATE = """You are a resume parsing engine. Extract structured information from the resume below.
Do NOT extract work experience - that is handled separately. Focus only on the fields below.

Return ONLY a JSON object with EXACTLY this structure (no extra text, no markdown, no explanations):

{schema}

CRITICAL: The structure above is ONLY an example of the JSON SHAPE (field names and types). The
placeholder text inside angle brackets like "<skill from text>" is NOT real data - you MUST ignore
it completely and extract the REAL values only from the actual resume text provided below. If a
field has nothing in the actual resume, return "" or [] for it - never copy the placeholder text,
and never invent realistic-sounding values that aren't explicitly in the resume.

Rules:
- education MUST always be a JSON array/list, even if there is only one entry:
  [{{"degree": "...", "institution": "...", "year": "..."}}]. NEVER return education as a single object.
- Education is not always phrased with words like "University" or "Bachelor". Lines like
  "Faculty of Artificial Intelligence - Third Year Student" or "Faculty of Engineering, Cairo University"
  ARE education entries. Put the faculty/major name in "degree" and any university name (if present) in
  "institution". Example: "Faculty of Artificial Intelligence - Third Year Student" ->
  {{"degree": "Faculty of Artificial Intelligence - Third Year Student", "institution": "", "year": ""}}
- education.year: if no graduation year is stated, use "" - do not guess or invent a year. If the
  resume states an expected/future graduation year (e.g. "Expected Graduation: 2026"), use that year.
- List ALL skills mentioned, do not omit any, even if the skills list is long.
- Do NOT include job titles or role names as skills (e.g. "Digital Marketing Specialist",
  "Software Engineer", "Marketing Manager" are job titles, not skills - they belong only in
  experience.title, never in the skills list).
- IMPORTANT boundary between skills and courses: anything listed under a "Skills" / "Technical
  Skills" / "Core Competencies" section belongs in "skills", even if its name sounds like a
  training program or software product (e.g. "Advanced SAP", "Database management software" are
  SKILLS, not courses, if they appear in a skills list).
- courses: ONLY standalone learning items/certifications with their own dedicated mention, typically
  named like "<Course Name> - <Platform>" (e.g. "Machine Learning - Coursera") or listed under a
  "Courses" / "Certifications" heading. Do NOT move ordinary skill-list entries into courses just
  because their name sounds technical. If none are mentioned, return an empty array [] - do not invent one.
- activities: extract extracurricular activities, clubs, student unions, competitions, or volunteering
  mentioned in the resume that are NOT formal work experience. A section literally titled
  "Volunteering" or "Volunteer Experience" belongs here in full, even if it mentions technical work
  (e.g. "Volunteered as Junior Developer in an app launch project" is an activity, not a job, unless
  it appears in the main Experience section). If none are mentioned, return an empty array [] - do
  not invent one.
- languages: only include languages explicitly mentioned in the resume. If none are mentioned, return [].
- projects: only include if the resume actually has a projects section. If there are no projects,
  return an empty array [] - do not invent a project.
- If a field is missing in the resume, use an empty string "" or empty list [].
- Do not invent any information not explicitly present in the resume text.

Resume text:
\"\"\"
{resume_text}
\"\"\"
"""


EXPERIENCE_SCHEMA_EXAMPLE = {
    "experience": [
        {"title": "<job title from text>", "company": "<company from text>",
         "duration": "<dates from text or empty>", "description": "<description from text>"}
    ]
}

EXPERIENCE_PROMPT_TEMPLATE = """You are a resume parsing engine. Your ONLY task is to extract work
experience entries from the resume text below. Ignore all other resume sections.

Return ONLY a JSON object with EXACTLY this structure (no extra text, no markdown, no explanations):

{schema}

CRITICAL: The structure above is ONLY an example of the JSON SHAPE. The placeholder text inside angle
brackets is NOT real data - ignore it completely. Extract REAL values only from the actual resume text
provided below. If there is genuinely no work experience, return {{"experience": []}} - never copy the
placeholder text or invent a realistic-sounding job that isn't in the resume.

Rules:
- A work experience entry is any job, internship, instructor role, or paid/professional position -
  NOT a personal project, NOT a course, NOT a student club/activity.
- Job entries are not always written in a clean "Job Title at Company (Start - End)" format. They are
  often written on ONE line with dashes, en-dashes, or pipes separating the parts.
- Dates can appear in many formats: "2022-2024", "(Jan 2022 - Present)", "2025–2026". Capture the
  duration exactly as written (you may normalize "–" to "-"), including words like "Present" or
  "Current" if used - do not leave duration empty just because it isn't a plain year range.
- Relative dates like "Since 2020" or "Since Jan 2020" mean the role started then and continues to
  the present - capture duration as "2020-Present" (or "Jan 2020-Present").
- If a resume lists multiple roles/titles under the SAME company (e.g. a promotion, such as
  "Junior Developer (2020-2021)" followed by "Senior Developer (2021-2023)" both at the same
  company), extract them as SEPARATE experience entries, each with its own title and duration, but
  repeating the same company name in each entry - do not merge them into a single entry or drop
  the earlier role.
- Some jobs are mentioned within a narrative sentence rather than a distinct line (e.g. "Worked
  briefly at Nile Insights doing freelance data entry, around 2018"). If a sentence names a specific
  company/client and describes paid work done there, extract it as an experience entry too, even
  though it's written as prose rather than a clear title/company/date line.
- Some entries give ONLY a company name and a year, with NO separate job title on its own (e.g.
  "Bloom Retail Co 2019"). This IS still a valid experience entry - extract company = "Bloom Retail
  Co", duration = "2019", and title = "" (empty) if no title is distinguishable. NEVER skip an entry
  just because the job title isn't clearly separated from the company name.
- The bullet points immediately following a job's title/company/duration line are that job's
  description - combine them into one "description" string.
- If NO duration is stated for a role, use "" for duration but still include the entry - do not skip
  an experience entry just because dates are missing.
- Do not include personal projects, courses, or extracurricular activities here, even if listed
  near the experience section.
- Do not invent any information not explicitly present in the resume text.

Here is a full worked EXAMPLE showing the expected pattern, including tricky cases like a
Present-dated role and a company-only line with no separate title. This example text is NOT the
resume you need to process - it only illustrates the format:

EXAMPLE INPUT:
\"\"\"
AI Instructor — International Future Window 2025–2026    Riyadh, Saudi Arabia
Delivered hands-on AI training to students covering Arduino, AI tools.

Marketing Specialist — GrowFast Agency, Cairo (Jan 2022 - Present)
Managed paid ad campaigns and increased engagement by 40%.

Bloom Retail Co 2019
Handled seasonal campaign planning and coordinated with the design team.
\"\"\"

EXAMPLE OUTPUT:
{{
  "experience": [
    {{"title": "AI Instructor", "company": "International Future Window", "duration": "2025-2026",
      "description": "Delivered hands-on AI training to students covering Arduino, AI tools."}},
    {{"title": "Marketing Specialist", "company": "GrowFast Agency, Cairo", "duration": "Jan 2022 - Present",
      "description": "Managed paid ad campaigns and increased engagement by 40%."}},
    {{"title": "", "company": "Bloom Retail Co", "duration": "2019",
      "description": "Handled seasonal campaign planning and coordinated with the design team."}}
  ]
}}

Apply this exact same pattern to the REAL resume text below.

Resume text:
\"\"\"
{resume_text}
\"\"\"
"""


KNOWN_SECTION_HEADERS = [
    "experience", "work experience", "professional experience", "employment history",
    "work history", "career history", "relevant experience",
    "projects", "personal projects", "academic projects", "portfolio",
    "graduation project", "capstone project", "final year project", "thesis",
    "education", "academic background", "qualifications",
    "technical skills", "skills", "core skills", "tools & skills", "tools and skills",
    "core competencies", "key skills",
    "certifications", "courses", "certifications & courses",
    "certifications & training", "training", "training & certifications",
    "certifications and training",
    "activities", "extracurricular activities", "volunteer work", "volunteering",
    "languages", "language skills",
    "summary", "professional summary", "objective", "profile", "about me",
    "other", "additional information", "other info", "interests", "hobbies", "references"
]

EXPERIENCE_HEADER_NAMES = [
    "experience", "work experience", "professional experience", "employment history",
    "work history", "career history", "relevant experience"
]

# Semantic fallback: instead of an ever-growing list of exact header
# strings, each section type has a short MEANING description. A candidate
# header line that doesn't exact-match anything above gets compared by
# embedding similarity against these descriptions - so a resume using a
# header we've never seen before (e.g. "My Journey", "Where I've Worked")
# still gets classified correctly, instead of silently breaking.
SECTION_CONCEPTS = {
    "experience": "work experience, employment history, jobs held",
    "projects": "personal or academic projects, portfolio work, graduation project",
    "education": "academic education, degree, university",
    "skills": "technical skills, tools, and technologies",
    "courses": "certifications, courses, and training programs completed",
    "activities": "volunteering and extracurricular activities",
    "languages": "spoken languages",
    "summary": "professional summary or personal objective statement",
}

_section_embedding_model = None
_section_concept_embeddings = None


def _get_section_embedding_model():
    global _section_embedding_model
    if _section_embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _section_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _section_embedding_model


def _looks_like_a_header(line):
    """Quick heuristic: real section headers are short and don't read like
    a sentence (no ending punctuation, not too many words)."""
    cleaned = line.strip()
    if not cleaned or len(cleaned) > 40:
        return False
    word_count = len(cleaned.split())
    if word_count > 5:
        return False
    if cleaned.endswith((".", ",", ";")):
        return False
    return True


def _classify_header_semantic(line, threshold=0.45):
    """
    Compares a candidate header line's meaning against each known section
    concept using embeddings, returning the best-matching category name
    if it's a confident enough match, else None. Only called on lines that
    already look header-shaped (see _looks_like_a_header), to avoid paying
    the embedding cost on every line of the resume.
    """
    global _section_concept_embeddings
    from sentence_transformers import util

    model = _get_section_embedding_model()

    if _section_concept_embeddings is None:
        _section_concept_embeddings = {
            key: model.encode(desc, convert_to_tensor=True)
            for key, desc in SECTION_CONCEPTS.items()
        }

    line_embedding = model.encode(line.strip(), convert_to_tensor=True)

    best_category, best_score = None, 0.0
    for category, concept_embedding in _section_concept_embeddings.items():
        score = util.cos_sim(line_embedding, concept_embedding).item()
        if score > best_score:
            best_category, best_score = category, score

    return best_category if best_score >= threshold else None


def _extract_section(text, section_names, semantic_category=None):
    """
    Structural section slicing: finds a standalone header line matching one
    of section_names (case-insensitive exact match - fast path), and
    returns everything up to the next known section header. If no exact
    match is found and semantic_category is provided, falls back to
    embedding-based header classification so a novel header phrasing
    (never seen before) still works instead of silently failing.
    Returns None if nothing usable is found, so callers can fall back to
    scanning the full text.
    """
    lines = text.split("\n")
    all_headers = {h.lower() for h in KNOWN_SECTION_HEADERS}
    target_set = {h.lower() for h in section_names}

    start_idx = None

    # fast path: exact match against the known header list
    for i, line in enumerate(lines):
        cleaned = line.strip().lower().rstrip(":")
        if cleaned in target_set:
            start_idx = i + 1
            break

    # semantic fallback: no exact match found, try meaning-based matching
    # on header-shaped lines only (cheap heuristic filters most lines out
    # before we pay the embedding cost)
    if start_idx is None and semantic_category:
        try:
            for i, line in enumerate(lines):
                if not _looks_like_a_header(line):
                    continue
                category = _classify_header_semantic(line)
                if category == semantic_category:
                    start_idx = i + 1
                    break
        except Exception as e:
            # embeddings unavailable or failed - degrade gracefully to "not found"
            print(f"Semantic section detection unavailable ({e}), using exact-match only")

    if start_idx is None:
        return None

    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        cleaned = lines[j].strip().lower().rstrip(":")
        if cleaned in all_headers and cleaned not in target_set:
            end_idx = j
            break
        # semantic boundary check too, so an unseen NEXT section header
        # (e.g. a novel "Projects" phrasing) still stops the slice correctly
        if semantic_category and _looks_like_a_header(lines[j]):
            try:
                category = _classify_header_semantic(lines[j])
                if category and category != semantic_category:
                    end_idx = j
                    break
            except Exception:
                pass

    section_text = "\n".join(lines[start_idx:end_idx]).strip()
    return section_text if section_text else None


def _salvage_truncated_json_array(raw_text, array_key):
    """
    Safety net for when the model's response gets cut off mid-object
    (e.g. hits a token limit before finishing the JSON). Instead of losing
    everything, this recovers whichever entries were fully written before
    the cutoff, by walking the raw text and tracking brace depth/string
    state manually. Returns a list (possibly partial) or None if nothing
    usable could be recovered.
    """
    key_marker = f'"{array_key}"'
    key_idx = raw_text.find(key_marker)
    if key_idx == -1:
        return None

    array_start = raw_text.find('[', key_idx)
    if array_start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    obj_start = None
    last_complete_end = None

    for i in range(array_start, len(raw_text)):
        ch = raw_text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == '{':
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and obj_start is not None:
                last_complete_end = i

    if last_complete_end is None:
        return None

    salvaged_str = raw_text[array_start:last_complete_end + 1] + "]"
    try:
        return json.loads(salvaged_str)
    except json.JSONDecodeError:
        return None


def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def extract_experience(resume_text, max_retries=3):
    """
    Dedicated, focused call for work experience only. Three layers of
    defense against the failure modes we've seen:
    1. Section slicing: if the resume has a clear "Experience" header, only
       that section is sent to the model - eliminating any chance it
       confuses Projects (or other sections) with real work experience.
    2. A generous num_predict budget, since experience descriptions can be
       long and a cut-off response is worse than a slow one.
    3. Truncation salvage: if the response still gets cut off, we recover
       whichever entries were completed before the cutoff instead of
       discarding everything.
    """
    experience_section = _extract_section(resume_text, EXPERIENCE_HEADER_NAMES, semantic_category="experience")
    text_to_send = experience_section if experience_section else resume_text

    prompt = EXPERIENCE_PROMPT_TEMPLATE.format(
        schema=json.dumps(EXPERIENCE_SCHEMA_EXAMPLE, indent=2, ensure_ascii=False),
        resume_text=text_to_send
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model="qwen2.5:3b",
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0, "num_predict": 2000}
            )
            result = response["message"]["content"].strip()
            data = json.loads(result)

            if isinstance(data, list):
                # in case the model still returns a bare array despite instructions
                return data

            experience = data.get("experience", [])
            return experience if isinstance(experience, list) else []

        except json.JSONDecodeError as e:
            salvaged = _salvage_truncated_json_array(result, "experience")
            if salvaged is not None:
                print(f"[Experience attempt {attempt + 1}] Response was truncated, "
                      f"recovered {len(salvaged)} complete entries")
                return salvaged

            last_error = e
            print(f"[Experience attempt {attempt + 1}] JSON parse failed: {e}")
            continue
        except Exception as e:
            last_error = e
            print(f"[Experience attempt {attempt + 1}] Error: {e}")
            continue

    print(f"Experience extraction failed after {max_retries} attempts: {last_error}")
    return []


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
                model="qwen2.5:3b",
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0, "num_predict": 2000}
            )
            result = response["message"]["content"].strip()
            data = json.loads(result)

            # experience is extracted in its own dedicated call for better accuracy
            data["experience"] = extract_experience(resume_text, max_retries)

            return _post_process(data, resume_text)

        except json.JSONDecodeError as e:
            last_error = e
            print(f"[Attempt {attempt + 1}] JSON parse failed: {e}")
            continue
        except Exception as e:
            last_error = e
            print(f"[Attempt {attempt + 1}] Error: {e}")
            continue

    # last resort: don't crash the whole API request over one resume that
    # completely failed to parse - return an empty-but-valid structure,
    # with experience still attempted independently since it's isolated
    print(f"Full resume extraction failed after {max_retries} attempts: {last_error}. "
          f"Returning empty structure so the request doesn't crash.")
    fallback_data = {"skills": [], "education": [], "projects": [], "languages": [], "courses": [], "activities": []}
    fallback_data["experience"] = extract_experience(resume_text, max_retries)
    return _post_process(fallback_data, resume_text)


def _merge_skill_lists(matcher_list, ai_list):
    """
    Combines two skill lists, removing duplicates (case-insensitive) and
    dropping AI phrases that are just a wordier restatement of a skill
    already found in canonical form by the matcher.
    """
    seen = {}

    for item in matcher_list:
        key = item.strip().lower()
        if key:
            seen[key] = item.strip()

    matcher_keys_lower = set(seen.keys())

    for item in ai_list:
        cleaned = item.strip()
        key = cleaned.lower()
        if not key or key in seen:
            continue

        is_redundant = any(
            canonical_key in key and canonical_key != key
            for canonical_key in matcher_keys_lower
        )
        if is_redundant:
            continue

        seen[key] = cleaned

    return sorted(seen.values())


COURSES_HEADER_NAMES = [
    "certifications", "courses", "certifications & courses",
    "certifications & training", "training", "training & certifications",
    "certifications and training"
]


def _extract_courses_fallback(resume_text):
    """
    Safety net for courses/certifications: if the resume has a standalone
    "Certifications" or "Courses" section, grab its lines directly. This
    catches entries the model sometimes drops (e.g. a certification listed
    with no other context around it).
    """
    section = _extract_section(resume_text, COURSES_HEADER_NAMES)
    if not section:
        return []
    return [line.strip() for line in section.split("\n") if line.strip()]


def _merge_simple_lists(*lists):
    """Merges multiple lists, removing case-insensitive duplicates, preserving order."""
    seen = {}
    for lst in lists:
        for item in lst:
            cleaned = item.strip() if isinstance(item, str) else str(item).strip()
            key = cleaned.lower()
            if key and key not in seen:
                seen[key] = cleaned
    return list(seen.values())


def _post_process(data, resume_text=""):
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
    ai_skills = []
    for s in skills_raw:
        if isinstance(s, str):
            ai_skills.append(s)
        elif isinstance(s, dict):
            ai_skills.append(s.get("name", ""))
    ai_skills = [s for s in ai_skills if s]

    # safety net: catch skills the model dropped from long lists (a known
    # failure mode of the small model) by also scanning the raw resume text
    matcher_skills = extract_all_skills(resume_text) if resume_text else []
    gliner_skills = extract_skills_with_gliner(resume_text) if resume_text else []
    skills = _merge_skill_lists(matcher_skills, ai_skills)
    skills = _merge_skill_lists(skills, gliner_skills)

    ai_courses = data.get("courses") or []
    courses_fallback = _extract_courses_fallback(resume_text) if resume_text else []
    courses = _merge_simple_lists(ai_courses, courses_fallback)

    return {
        "skills": skills,
        "education": education,
        "experience": data.get("experience") or [],
        "projects": data.get("projects") or [],
        "languages": data.get("languages") or [],
        "courses": courses,
        "activities": data.get("activities") or []
    }


if __name__ == "__main__":
    import fitz
    doc = fitz.open(r"E:\ai-recruitment\data\resumes\Haneen_Elabd_CV.pdf")
    text = "\n".join(page.get_text() for page in doc)
    result = extract_resume_data(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))