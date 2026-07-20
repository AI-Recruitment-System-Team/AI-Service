import json


def normalize(text):
    return text.lower().replace(" basics", "").strip()


def match_resume_job(resume_data, job_data):

    resume_skills = resume_data.get("skills", [])
    job_skills = job_data.get("required_skills", [])

    matched = []
    missing = []

    for job_skill in job_skills:

        job_name = normalize(job_skill["name"])
        job_category = job_skill.get("category", "")

        found = False

        for resume_skill in resume_skills:

            resume_name = normalize(resume_skill["name"])
            resume_category = resume_skill.get("category", "")

            # Match by name
            if job_name in resume_name or resume_name in job_name:

                found = True

            # Match by similar category
            elif job_category == resume_category:
                
                if job_name.split()[0] in resume_name:
                    found = True


        if found:
            matched.append(job_skill["name"])
        else:
            missing.append(job_skill["name"])


    total = len(job_skills)

    score = 0

    if total:
        score = round((len(matched) / total) * 100)


    return {
        "match_score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "total_required_skills": total
    }



# ==========================
# Test
# ==========================

if __name__ == "__main__":

    resume = {
        "skills": [
            {
                "name": "Python",
                "category": "Programming Language"
            },
            {
                "name": "Machine Learning basics",
                "category": "Artificial Intelligence"
            },
            {
                "name": "Data Analysis",
                "category": "Data Science"
            },
            {
                "name": "Computer Vision basics",
                "category": "Artificial Intelligence"
            }
        ]
    }


    job = {
        "required_skills": [
            {
                "name": "Python",
                "category": "Programming Language"
            },
            {
                "name": "Machine Learning",
                "category": "Artificial Intelligence"
            },
            {
                "name": "TensorFlow",
                "category": "Framework or Library"
            },
            {
                "name": "SQL",
                "category": "Database"
            }
        ]
    }


    result = match_resume_job(resume, job)


    print(json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    ))