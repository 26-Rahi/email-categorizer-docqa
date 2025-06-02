# email-categorizer-docqa
# Email Categorizer & Document QA (Django + Transformers)

This Django-based web app automatically categorizes and summarizes emails using NLP, and also answers questions from uploaded documents (PDF/image) using OCR and document QA models.
# Features

-  *Email Categorization* using Zero-Shot Classification + Clustering (KMeans)
-  *Email Summarization* using BART-based summarizer
-  *Document Question Answering* from PDF or images using LayoutLM and OCR
-   Uses HuggingFace Transformers, Tesseract OCR, and Sklearn
-   Saves categorized email history to DB

# Tech Stack

- Python, Django
- Transformers: facebook/bart-large-mnli, distilbart-cnn-12-6, layoutlm-document-qa
- Sklearn: KMeans, TfidfVectorizer
- OCR: Tesseract + Poppler (for PDF images)
- Frontend: HTML/CSS

---

