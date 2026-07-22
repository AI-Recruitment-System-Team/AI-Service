import fitz  # PyMuPDF
import re


def extract_text_from_pdf(file_path):
    """
    Extracts full text from a PDF file using PyMuPDF.
    More reliable than pypdf at preserving text order, especially with
    multi-column layouts.
    """
    doc = fitz.open(file_path)

    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text += page_text + "\n"

    doc.close()

    return clean_text(text)


def clean_text(text):
    """
    Basic cleanup of extracted text:
    - collapse excessive blank lines
    - collapse repeated spaces/tabs
    """
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else r"E:\ai-recruitment\data\resumes\Haneen_Elabd_CV.pdf"
    print(extract_text_from_pdf(path))