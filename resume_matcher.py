from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

def read_file(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    except:
        print(f"Error: Cannot read file → {path}")
        sys.exit(1)

resume_text = read_file("resume.txt")
job_text = read_file("job_description.txt")

vectorizer = TfidfVectorizer(stop_words="english")
vectors = vectorizer.fit_transform([resume_text, job_text])

similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

print(f"Match Score: {similarity * 100:.2f}%")

