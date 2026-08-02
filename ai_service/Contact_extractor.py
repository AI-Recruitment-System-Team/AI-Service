import re


def extract_stated_years_of_experience(text):
    """
    Looks for an explicit years-of-experience phrase written in the resume
    (e.g. "6 years of experience", "over six years", "5+ years"). This is
    meant as a FALLBACK ONLY, used by the matching engine when calculating
    years from actual job dates yields zero (e.g. no dates were found at
    all) - actual job dates are always preferred when available, since a
    self-stated number can be rounded or exaggerated.
    Returns an int, or None if no such phrase is found.
    """
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }

    # numeric form: "6 years of experience", "5+ years", "over 6 years"
    match = re.search(r"(\d+)\+?\s*years?\s*(?:of)?\s*experience", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # word form: "six years of experience", "over six years"
    words_pattern = "|".join(number_words.keys())
    match = re.search(
        rf"\b({words_pattern})\b\s*years?\s*(?:of)?\s*experience",
        text, re.IGNORECASE
    )
    if match:
        return number_words[match.group(1).lower()]

    return None


def extract_name(text):
    """
    The first non-empty line without @ or linkedin/github is usually the name.
    """
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if len(line) > 2 and "@" not in line and "linkedin" not in line.lower() and "github" not in line.lower():
            return line
    return ""


def extract_email(text):
    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )
    return match.group(0) if match else ""


def extract_phone(text):
    # Supports Egyptian and international formats (+20, 00, or a plain 10-13 digit number)
    match = re.search(
        r"(\+?\d{1,3}[\s-]?)?(\(?\d{2,4}\)?[\s-]?){2,4}\d{3,4}",
        text
    )
    if match:
        digits = re.sub(r"[^\d+]", "", match.group(0))
        if len(digits) >= 9:
            return digits
    return ""


def extract_linkedin(text):
    match = re.search(r"https?://(?:[a-z]{2}\.)?linkedin\.com/\S+", text, re.IGNORECASE)
    return match.group(0) if match else ""


def extract_github(text):
    match = re.search(r"https?://(?:www\.)?github\.com/\S+", text, re.IGNORECASE)
    return match.group(0) if match else ""


def extract_portfolio(text):
    for match in re.finditer(r"https?://\S+", text, re.IGNORECASE):
        url = match.group(0)
        if "linkedin" not in url.lower() and "github" not in url.lower():
            return url
    return ""


_COMPANY_SUFFIX_WORDS = {
    "agency", "inc", "llc", "ltd", "co", "corp", "corporation", "group",
    "company", "solutions", "technologies", "studio", "studios", "labs",
    "systems", "partners", "consulting", "enterprises",
}


def extract_location(text):
    """
    Not restricted to Egypt - looks for a "City, Country" pattern in the
    first few lines of the CV (usually the header area). Requires both
    parts to be Title Case (like real place names: "Tanta, Egypt"),
    which prevents matching random lowercase prose fragments that happen
    to contain a comma (e.g. "deploying intelligent, autonomous systems").
    Also skips matches where the part before the comma looks like a
    company name (e.g. "BrightWave Agency, Cairo") rather than a city.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()][:5]
    header_chunk = " | ".join(lines)

    pattern = re.compile(
        r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*),\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b"
    )

    for match in pattern.finditer(header_chunk):
        before_comma = match.group(1)
        last_word = before_comma.split()[-1].lower()

        if last_word in _COMPANY_SUFFIX_WORDS:
            continue  # looks like "X Agency, City" - skip, keep searching

        location = match.group(0).split("|")[0].strip()
        if "@" not in location and "http" not in location:
            return location

    return ""


def extract_contact(text):
    return {
        "name": extract_name(text),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "location": extract_location(text),
        "links": {
            "linkedin": extract_linkedin(text),
            "github": extract_github(text),
            "portfolio": extract_portfolio(text)
        }
    }


if __name__ == "__main__":
    sample = """
    Haneen Elabd
    Tanta, Egypt | haneenelabd06@gmail.com | 01284980782
    LinkedIn: https://eg.linkedin.com/in/haneen-elabd-837b5532b
    """
    print(extract_contact(sample))