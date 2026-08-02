import os
import json
from cv_parser import parse_cv

RESUMES_FOLDER = r"E:\ai-recruitment\data\resumes"
OUTPUT_FOLDER = r"E:\ai-recruitment\data\parsed_resumes"


def process_folder(resumes_folder=RESUMES_FOLDER, output_folder=OUTPUT_FOLDER):
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(resumes_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in: {resumes_folder}")
        return

    print(f"Processing {len(pdf_files)} file(s)...\n")

    success_count = 0
    failed_files = []

    for filename in pdf_files:
        file_path = os.path.join(resumes_folder, filename)
        print(f"-> Processing: {filename}")

        try:
            data = parse_cv(file_path)

            output_name = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(output_folder, output_name)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            print(f"   saved -> {output_path}\n")
            success_count += 1

        except Exception as e:
            print(f"   FAILED: {e}\n")
            failed_files.append(filename)

    print("=" * 40)
    print(f"Done: {success_count}/{len(pdf_files)} succeeded")
    if failed_files:
        print("Failed:")
        for f in failed_files:
            print(f"  - {f}")


if __name__ == "__main__":
    process_folder()