import re


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


def extract_location(text):
    """
    Not restricted to Egypt - looks for a "City, Country" pattern in the
    first few lines of the CV (usually the header area).
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()][:5]
    header_chunk = " | ".join(lines)

    # generic pattern: word(s) + comma + word(s) (city/governorate, country)
    match = re.search(r"([A-Za-z][A-Za-z\s]{2,20},\s*[A-Za-z][A-Za-z\s]{2,20})", header_chunk)
    if match:
        location = match.group(1).split("|")[0].strip()
        # avoid accidentally matching part of an email or URL
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