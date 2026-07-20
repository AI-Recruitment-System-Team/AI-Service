import re


def extract_contact(header_text):

    data = {
        "name": "",
        "email": "",
        "phone": "",
        "location": "",
        "links": {
            "linkedin": "",
            "github": "",
            "portfolio": ""
        }
    }


    lines = [
        line.strip()
        for line in header_text.split("\n")
        if line.strip()
    ]


    # -------- Name --------
    if lines:
        data["name"] = lines[0]


    # -------- Email --------
    email = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        header_text
    )

    if email:
        data["email"] = email.group(0)


    # -------- Phone --------
    phone = re.search(
        r"(\+?\d{10,13})",
        header_text
    )

    if phone:
        data["phone"] = phone.group(0)


    # -------- LinkedIn --------
    linkedin = re.search(
        r"https?://[^\s]*linkedin[^\s]*",
        header_text,
        re.IGNORECASE
    )

    if linkedin:
        data["links"]["linkedin"] = linkedin.group(0)


    # -------- GitHub --------
    github = re.search(
        r"https?://[^\s]*github[^\s]*",
        header_text,
        re.IGNORECASE
    )

    if github:
        data["links"]["github"] = github.group(0)


    # -------- Portfolio --------
    portfolio = re.search(
        r"https?://(?!.*(?:linkedin|github))[^\s]+",
        header_text,
        re.IGNORECASE
    )

    if portfolio:
        data["links"]["portfolio"] = portfolio.group(0)



    # -------- Location --------
    # غالباً السطر الثاني يحتوي المكان
    if len(lines) > 1:

        location_line = lines[1]

        location = location_line.split("|")[0].strip()

        if location:
            data["location"] = location



    return data



# -----------------------
# Test
# -----------------------

if __name__ == "__main__":


    header = """
    Haneen Elabd
    Tanta, Egypt | haneenelabd06@gmail.com | 01284980782
    LinkedIn: https://eg.linkedin.com/in/haneen-elabd-837b5532b
    """


    result = extract_contact(header)


    print(result)