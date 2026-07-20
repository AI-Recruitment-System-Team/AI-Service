def validate_cv(data):

    validated = {
        "name": data.get("name"),
        "email": data.get("email"),
        "phone": data.get("phone"),

        "skills": data.get("skills", []),

        "education": {
            "institution": "",
            "degree": "",
            "year": ""
        },

        "experience": data.get("experience", []),
        "projects": data.get("projects", []),
        "languages": data.get("languages", [])
    }

    education = data.get("education", {})

    if isinstance(education, dict):

        validated["education"]["institution"] = (
            education.get("institution")
            or education.get("faculty")
            or education.get("facultyName")
            or education.get("university")
            or ""
        )

        validated["education"]["degree"] = (
            education.get("degree")
            or ""
        )

        validated["education"]["year"] = (
            education.get("year")
            or education.get("year_of_study")
            or ""
        )

    return validated