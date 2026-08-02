import ollama
import json
import re
from skills_matcher import extract_skills_with_requirement_level

SCHEMA_EXAMPLE = {
    "job_title": "<extract from text>",
    "company": "<extract from text or empty string>",
    "location": "<extract from text or empty string>",
    "employment_type": "<e.g. Full-time / Part-time / Internship / Remote, or empty string>",
    "seniority_level": "<e.g. Junior / Mid-level / Senior, or empty string>",
    "years_of_experience": "<e.g. 2+ years, or empty string>",
    "required_skills": ["<skill from text>", "..."],
    "nice_to_have_skills": ["<optional skill from text>", "..."],
    "education_requirements": ["<requirement from text>", "..."],
    "responsibilities": ["<duty from text>", "..."],
    "languages_required": ["<language from text>", "..."],
    "salary_range": "<extract from text or empty string>",
    "benefits": ["<benefit from text>", "..."]
}

PROMPT_TEMPLATE = """You are a job description parsing engine. Extract structured information
from the job description text below. It can be for ANY role or industry (tech, marketing,
sales, healthcare, education, finance, etc.) and in ANY format (bullet points, paragraphs,
LinkedIn-style posts, plain announcements).

Return ONLY a JSON object with EXACTLY this structure (no extra text, no markdown, no explanations):

{schema}

CRITICAL: The structure above is ONLY an example of the JSON SHAPE (field names and types).
The values inside it (like "Backend Developer", "XYZ Corp", "2-4 years") are FAKE placeholder
values, NOT the answer. You MUST ignore these example values completely and extract the REAL
values only from the actual job description text provided below. If a field is not mentioned
in the actual text, return "" or [] for it — do NOT reuse any example value.

Rules:
- company: only extract an ACTUAL company/organization name if one is explicitly named (e.g. "XYZ
  Corp is hiring..."). Do NOT extract pronouns or generic phrases like "We", "Our team", "I" as a
  company name just because the JD starts with "We're looking for..." or similar casual phrasing.
  If no specific company name is given anywhere in the text, return "" - a vague opening like
  "We're hiring" does NOT count as a company name.
- required_skills / nice_to_have_skills: each entry must be a SHORT atomic skill/tool/technology
  name only (e.g. "Python", "PostgreSQL", "Docker") — NEVER a full sentence or a requirement
  description. If a line mentions multiple skills together (e.g. "SQL databases (PostgreSQL
  preferred)"), split it into separate entries: "SQL", "PostgreSQL".
- Degree/education requirements (e.g. "Bachelor's degree in Computer Science") NEVER belong in
  required_skills or nice_to_have_skills — they always belong in education_requirements only.
- required_skills: skills/tools/technologies explicitly stated as required, must-have, or listed
  under a "Requirements" / "Qualifications" section without a "preferred"/"nice to have"/"plus"
  qualifier.
- nice_to_have_skills: skills explicitly marked as preferred, a plus, bonus, or nice to have.
  Same atomic-term rule applies (e.g. "Docker", "AWS", not "Experience with Docker and AWS").
  If the JD doesn't distinguish between required and optional skills, put everything under
  required_skills and leave nice_to_have_skills as [].
- languages_required: this means HUMAN/SPOKEN languages only (e.g. "English", "Arabic", "French",
  "Mandarin"). NEVER put programming languages, frameworks, or technical skills here (e.g. "C#",
  "SQL", "JavaScript" are technical skills and belong in required_skills, NOT languages_required).
  If no spoken language requirement is explicitly mentioned, return an empty array [].
- years_of_experience: extract exactly as stated (e.g. "3+ years", "2-4 years", "entry-level").
  If not mentioned, use "".
- seniority_level: infer only from explicit wording (e.g. "Senior", "Junior", "Entry-level",
  "Lead", "Mid-level"). If not stated or implied clearly, use "".
- education_requirements: only include if explicitly mentioned (e.g. degree, major, certification
  requirement). Do not invent a requirement if the JD doesn't mention one.
- employment_type: e.g. "Full-time", "Part-time", "Internship", "Contract", "Remote", "Hybrid",
  "On-site". Use "" if not mentioned.
- responsibilities: list the core duties/tasks of the role, in the resume's own wording,
  shortened to concise bullet phrases.
- salary_range: only if a number or range is explicitly mentioned. Otherwise "".
- If a field is missing from the text, use "" or [] as appropriate. Do not invent any information.
- Do not confuse company benefits/perks with job responsibilities.

Here is a full worked EXAMPLE showing exactly how a similar job description should be converted.
This example text is NOT the one you need to process — it only shows the expected pattern:

EXAMPLE INPUT:
\"\"\"
Requirements:
- 3+ years of experience with JavaScript and React
- Familiarity with NoSQL databases (MongoDB preferred)
- BSc in Computer Science or a related field

Nice to have: TypeScript, GraphQL
\"\"\"

EXAMPLE OUTPUT:
{{
  "required_skills": ["JavaScript", "React", "NoSQL"],
  "nice_to_have_skills": ["MongoDB", "TypeScript", "GraphQL"],
  "education_requirements": ["BSc in Computer Science or a related field"],
  "years_of_experience": "3+ years"
}}

Notice: "MongoDB preferred" inside a parenthesis means the general term (NoSQL) is required but
the specific preferred one (MongoDB) goes to nice_to_have_skills. The degree requirement goes to
education_requirements, never to skills. Apply this exact same pattern to the REAL job description
below.

Job description text:
\"\"\"
{jd_text}
\"\"\"
"""


def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def extract_jd_data(jd_text, max_retries=3):
    jd_text = clean_text(jd_text)

    prompt = PROMPT_TEMPLATE.format(
        schema=json.dumps(SCHEMA_EXAMPLE, indent=2, ensure_ascii=False),
        jd_text=jd_text
    )

    last_error = None
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model="qwen2.5:3b",
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0, "num_predict": 1000}
            )
            result = response["message"]["content"].strip()
            data = json.loads(result)
            processed = _post_process(data)
            return _merge_with_skills_matcher(processed, jd_text)

        except json.JSONDecodeError as e:
            last_error = e
            print(f"[Attempt {attempt + 1}] JSON parse failed: {e}")
            continue
        except Exception as e:
            last_error = e
            print(f"[Attempt {attempt + 1}] Error: {e}")
            continue

    raise ValueError(f"Failed after {max_retries} attempts: {last_error}")


def _merge_skill_lists(matcher_list, ai_list):
    """
    Combines two skill lists, removing duplicates (case-insensitive) and
    also dropping AI phrases that are just a wordier restatement of a skill
    already found in canonical form by the matcher
    (e.g. "CI/CD pipelines" is dropped because "CI/CD" is already present).
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

        # skip if this AI phrase merely contains an already-known canonical
        # skill as a substring (e.g. "CI/CD pipelines" contains "ci/cd")
        is_redundant = any(
            canonical_key in key and canonical_key != key
            for canonical_key in matcher_keys_lower
        )
        if is_redundant:
            continue

        seen[key] = cleaned

    return sorted(seen.values())


def _merge_with_skills_matcher(data, jd_text):
    """
    Runs the keyword-based skills_matcher.py alongside the AI result and
    merges them. This catches skills the AI model missed or mangled
    (e.g. skills buried in parentheses), while still keeping anything
    the AI found that isn't in the fixed skills list.
    """
    matcher_result = extract_skills_with_requirement_level(jd_text)

    merged_required = _merge_skill_lists(
        matcher_result["required"], data["required_skills"]
    )
    merged_nice_to_have = _merge_skill_lists(
        matcher_result["nice_to_have"], data["nice_to_have_skills"]
    )

    # a skill required somewhere shouldn't also appear as nice-to-have
    required_lower = {s.lower() for s in merged_required}
    merged_nice_to_have = [s for s in merged_nice_to_have if s.lower() not in required_lower]

    data["required_skills"] = merged_required
    data["nice_to_have_skills"] = merged_nice_to_have
    return data


def _post_process(data):
    """Normalize field types so downstream code (matching) can always rely on the same shape."""

    def as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        return []

    def as_str(value):
        if value is None:
            return ""
        return str(value).strip()

    _INVALID_COMPANY_VALUES = {"we", "our", "our team", "us", "i", "you", "they", "the company"}

    def as_company(value):
        cleaned = as_str(value)
        if cleaned.lower() in _INVALID_COMPANY_VALUES:
            return ""
        return cleaned

    return {
        "job_title": as_str(data.get("job_title")),
        "company": as_company(data.get("company")),
        "location": as_str(data.get("location")),
        "employment_type": as_str(data.get("employment_type")),
        "seniority_level": as_str(data.get("seniority_level")),
        "years_of_experience": as_str(data.get("years_of_experience")),
        "required_skills": as_list(data.get("required_skills")),
        "nice_to_have_skills": as_list(data.get("nice_to_have_skills")),
        "education_requirements": as_list(data.get("education_requirements")),
        "responsibilities": as_list(data.get("responsibilities")),
        "languages_required": as_list(data.get("languages_required")),
        "salary_range": as_str(data.get("salary_range")),
        "benefits": as_list(data.get("benefits"))
    }


if __name__ == "__main__":
    sample_jd = """
    We are hiring a Backend Developer to join our growing team in Cairo.

    Requirements:
    - 2+ years of experience with Python and Django
    - Strong knowledge of SQL databases (PostgreSQL preferred)
    - Bachelor's degree in Computer Science or related field

    Nice to have:
    - Experience with Docker and AWS
    - Familiarity with CI/CD pipelines

    Responsibilities:
    - Design and maintain REST APIs
    - Collaborate with the frontend team
    - Write unit tests and documentation

    Benefits: Health insurance, flexible hours, remote work option.
    """

    result = extract_jd_data(sample_jd)
    print(json.dumps(result, indent=2, ensure_ascii=False))