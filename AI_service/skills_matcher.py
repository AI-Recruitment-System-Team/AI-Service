import re
import json
import os

# =====================================================================
# SKILLS TAXONOMY
# Loaded from skills_taxonomy.json (same folder as this file) so anyone
# can add a new skill by editing a plain JSON file - no code changes,
# no need to touch Python at all.
# =====================================================================

_TAXONOMY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills_taxonomy.json")

with open(_TAXONOMY_PATH, encoding="utf-8") as _f:
    SKILLS_TAXONOMY = json.load(_f)


# Phrases that mark a skill as optional/preferred when found inline,
# in the same segment as the skill mention (e.g. inside a parenthesis).
NICE_TO_HAVE_INLINE_MARKERS = [
    "nice to have", "nice-to-have", "preferred", "a plus", "is a plus",
    "bonus", "optional", "desirable", "would be a plus", "plus if",
    "not required but",
]

# Standalone section headers (e.g. "Nice to have:") that switch the
# classification of every bullet point underneath them.
NICE_TO_HAVE_HEADERS = {
    "nice to have", "nice-to-have", "preferred", "preferred qualifications",
    "preferred skills", "bonus", "bonus points", "optional", "desirable",
    "good to have", "would be a plus", "pluses",
}

REQUIRED_HEADERS = {
    "requirements", "required", "required skills", "must have", "must-have",
    "qualifications", "responsibilities", "duties", "required qualifications",
    "minimum qualifications", "core skills",
}


def _classify_header(line):
    """
    Detects standalone section headers like "Nice to have:" or "Requirements:"
    that switch the classification of every bullet point underneath them,
    even though the bullet points themselves don't repeat the marker word.
    Returns "nice", "required", or None if the line isn't a recognized header.
    """
    cleaned = line.strip().rstrip(":").strip().lower()
    if cleaned in NICE_TO_HAVE_HEADERS:
        return "nice"
    if cleaned in REQUIRED_HEADERS:
        return "required"
    return None


def _build_patterns():
    """
    Compiles one regex per (canonical skill, alias) pair.
    Sorted by term length (longest first) so multi-word terms like
    "Machine Learning" are matched before any shorter overlapping term.
    """
    entries = []
    for canonical, aliases in SKILLS_TAXONOMY.items():
        terms = [canonical] + aliases
        for term in terms:
            entries.append((canonical, term))

    entries.sort(key=lambda x: len(x[1]), reverse=True)

    compiled = []
    for canonical, term in entries:
        pattern = re.compile(
            r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])",
            re.IGNORECASE
        )
        compiled.append((canonical, pattern))

    return compiled


_COMPILED_PATTERNS = _build_patterns()


def _line_has_inline_marker(segment_text):
    lower = segment_text.lower()
    return any(marker in lower for marker in NICE_TO_HAVE_INLINE_MARKERS)


def _find_matches_no_overlap(text):
    """
    Finds all skill matches in a text segment, resolving overlaps so a
    shorter term (e.g. "C") doesn't get matched inside a longer one
    (e.g. "C++") that starts at the same position.
    """
    all_matches = []
    for canonical, pattern in _COMPILED_PATTERNS:
        for m in pattern.finditer(text):
            all_matches.append((m.start(), m.end(), canonical))

    # longest matches win; process longest-first and skip anything that
    # overlaps a span we've already accepted
    all_matches.sort(key=lambda x: (x[1] - x[0]), reverse=True)

    accepted = []
    occupied = []
    for start, end, canonical in all_matches:
        overlaps = any(not (end <= s or start >= e) for s, e in occupied)
        if not overlaps:
            accepted.append(canonical)
            occupied.append((start, end))

    return accepted


def _split_line_into_scopes(line):
    """
    Splits a line into (text_segment, has_inline_marker) scopes, treating
    parenthetical asides as their own scope. This correctly handles patterns
    like "SQL databases (PostgreSQL preferred)" where the marker ("preferred")
    should only apply to what's inside the parentheses, not the whole line.
    """
    scopes = []
    last_end = 0

    for m in re.finditer(r"\(([^)]*)\)", line):
        outside = line[last_end:m.start()]
        if outside.strip():
            scopes.append((outside, _line_has_inline_marker(outside)))

        inside = m.group(1)
        if inside.strip():
            scopes.append((inside, _line_has_inline_marker(inside)))

        last_end = m.end()

    remainder = line[last_end:]
    if remainder.strip():
        scopes.append((remainder, _line_has_inline_marker(remainder)))

    if not scopes:
        scopes = [(line, _line_has_inline_marker(line))]

    return scopes


def extract_all_skills(text):
    """
    Returns a flat, deduplicated, sorted list of every known skill found
    in the text. Use this for CVs, where required/nice-to-have doesn't apply.
    """
    found = set()
    for line in text.split("\n"):
        found.update(_find_matches_no_overlap(line))
    return sorted(found)


def extract_skills_with_requirement_level(text):
    """
    Returns {"required": [...], "nice_to_have": [...]}.
    Use this for job descriptions. Classification uses two signals:
    1. Section state: a standalone header line like "Nice to have:" switches
       every bullet point below it to optional, until a "Requirements:"-style
       header switches it back.
    2. Inline markers: within a line, parenthetical asides like
       "(PostgreSQL preferred)" are judged independently of the rest of the line.
    A skill is optional if EITHER signal says so.
    """
    required = set()
    nice_to_have = set()
    section_is_nice = False

    for line in text.split("\n"):
        if not line.strip():
            continue

        header = _classify_header(line)
        if header == "nice":
            section_is_nice = True
            continue
        if header == "required":
            section_is_nice = False
            continue

        for segment, has_inline_marker in _split_line_into_scopes(line):
            is_nice = section_is_nice or has_inline_marker
            matches = _find_matches_no_overlap(segment)
            for canonical in matches:
                if is_nice:
                    nice_to_have.add(canonical)
                else:
                    required.add(canonical)

    # a skill explicitly required somewhere should not also sit in nice_to_have
    nice_to_have -= required

    return {
        "required": sorted(required),
        "nice_to_have": sorted(nice_to_have)
    }


if __name__ == "__main__":
    sample_jd = """
    Requirements:
    - 2+ years of experience with Python and Django
    - Strong knowledge of SQL databases (PostgreSQL preferred)
    - Bachelor's degree in Computer Science or related field

    Nice to have:
    - Experience with Docker and AWS
    - Familiarity with CI/CD pipelines
    """

    print("JD skills:", extract_skills_with_requirement_level(sample_jd))

    sample_cv = "Skills: Python, C++, Machine Learning basics, Data Analysis, Computer Vision basics, Web Development"
    print("CV skills:", extract_all_skills(sample_cv))