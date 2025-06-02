from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from .models import EmailHistory
import os
os.environ['TRANSFORMERS_CACHE'] = 'D:/transformers_cache'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from django.conf import settings

# Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

# Load summarizer & QA pipeline
summarizer_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
qa_pipeline = pipeline("question-answering", model="impira/layoutlm-document-qa")

# Zero-shot classifier (dynamic categorization)
zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Cluster label mappings (customize as needed)
cluster_labels = {
    0: "Work",
    1: "Spam",
    2: "Personal",
    3: "Updates"
}

# --- Helpers ---

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    filtered = [w for w in tokens if w not in stop_words]
    return " ".join(filtered)

def summarize_text(text):
    inputs = summarizer_tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = summarizer_model.generate(
        inputs, max_length=100, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    return summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=r'D:\poppler-24.08.0\Library\bin')
    full_text = ""
    for image in images:
        text = pytesseract.image_to_string(image)
        full_text += text + "\n"
    return full_text

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def ask_document_question(file_path, question, file_type):
    if file_type == 'pdf':
        context = extract_text_from_pdf(file_path)
    else:
        context = extract_text_from_image(file_path)
    result = qa_pipeline(question=question, context=context)
    return result['answer'], result['score']

def handle_uploaded_file(f):
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f.name)
    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return file_path

def cluster_emails_dynamic(all_texts, num_clusters=4):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(all_texts)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters, kmeans

# --- Main View ---

def index(request):
    category = ""
    summarized_text = ""
    email_text = ""
    answer = None
    confidence = None

    if request.method == "POST":
        email_text = request.POST.get("email_text")
        uploaded_file = request.FILES.get('pdf_file')
        question = request.POST.get("question")

        if "summarize" in request.POST:
            summarized_text = summarize_text(email_text)

        elif "categorize" in request.POST:
            # Zero-shot classification
            candidate_labels = ["Work", "Personal", "Spam", "Updates"]
            zero_shot_result = zero_shot_classifier(email_text, candidate_labels)
            zero_shot_category = zero_shot_result['labels'][0]  # Top label from zero-shot

            # Get all emails from DB + new one for clustering
            all_emails = [email.content for email in EmailHistory.objects.all()]
            all_emails.append(email_text)

            # Perform clustering
            clusters, _ = cluster_emails_dynamic(all_emails, num_clusters=4)
            current_cluster = clusters[-1]  # Cluster assigned to new email
            cluster_category = cluster_labels.get(current_cluster, f"Cluster-{current_cluster}")

            # Decide final category — here overriding zero-shot with cluster result
            category = cluster_category

            # Save email with final category
            EmailHistory.objects.create(content=email_text, category=category)

        elif uploaded_file and question:
            file_name = uploaded_file.name
            extension = file_name.split('.')[-1].lower()
            file_type = 'pdf' if extension == 'pdf' else 'image'
            save_path = handle_uploaded_file(uploaded_file)
            try:
                answer, confidence = ask_document_question(save_path, question, file_type)
            except Exception as e:
                answer = f"Error processing document: {str(e)}"
                confidence = 0

    return render(request, "categorizer/index.html", {
        "category": category,
        "summarized_text": summarized_text,
        "email_text": email_text,
        "answer": answer,
        "confidence": confidence
    })

# --- History View ---

def history(request):
    emails = EmailHistory.objects.all().order_by('-created_at')
    return render(request, "categorizer/history.html", {"emails": emails})