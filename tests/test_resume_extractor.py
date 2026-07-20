from resume_extractor import extract_resume_data


sections = {
    "Objective": """
    Computer Science student specializing in Artificial Intelligence seeking an internship opportunity.
    """,

    "Education": """
    Faculty of Artificial Intelligence – Third Year Student
    """,

    "Technical Skills": """
    Python, C++, Machine Learning basics, Data Analysis,
    Computer Vision basics, Web Development
    """,

    "Projects": """
    Clinic Booking Website: Web application for booking doctor appointments.
    Line Follower Robot: Built an autonomous robot using sensors and control logic.
    Calorie Calculator ML Project: Machine learning model for calorie estimation.
    """,

    "Courses": """
    Machine Learning – Coursera
    """,

    "Activities": """
    Student Union Member, Startup Hub Kafr El-Sheikh participant
    """,

    "Languages": """
    Arabic (Native), English (Good)
    """
}


result = extract_resume_data(sections)


print(result)