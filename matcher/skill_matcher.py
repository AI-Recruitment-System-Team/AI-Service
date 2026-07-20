from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# تحميل موديل Embedding صغير
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_skill_embedding(skill):
    return model.encode(skill)


def calculate_skill_similarity(resume_skills, job_skills):

    matched = []
    missing = []

    score_sum = 0


    for job_skill in job_skills:

        best_score = 0
        best_match = None

        job_vector = get_skill_embedding(job_skill["name"])


        for resume_skill in resume_skills:

            resume_vector = get_skill_embedding(resume_skill["name"])

            similarity = cosine_similarity(
                [job_vector],
                [resume_vector]
            )[0][0]


            if similarity > best_score:
                best_score = similarity
                best_match = resume_skill["name"]


        # threshold
        if best_score >= 0.65:

            matched.append({
                "required": job_skill["name"],
                "matched_with": best_match,
                "similarity": round(float(best_score),2)
            })

            score_sum += best_score

        else:

            missing.append(job_skill["name"])



    final_score = 0

    if job_skills:
        final_score = round(
            (score_sum / len(job_skills)) * 100
        )


    return {
        "skill_score": final_score,
        "matched_skills": matched,
        "missing_skills": missing
    }
if __name__ == "__main__":

    resume_skills = [
        {"name": "Python"},
        {"name": "Machine Learning"},
        {"name": "Data Analysis"},
        {"name": "OpenCV"}
    ]

    job_skills = [
        {"name": "Python"},
        {"name": "Machine Learning"},
        {"name": "TensorFlow"},
        {"name": "SQL"},
        {"name": "Computer Vision"}
    ]

    result = calculate_skill_similarity(
        resume_skills,
        job_skills
    )

    from pprint import pprint
    pprint(result)