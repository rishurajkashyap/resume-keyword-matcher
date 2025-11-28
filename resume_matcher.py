from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resume_text = open("resume.txt", "r").read()
job_text = open("job_description.txt", "r").read()

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([resume_text, job_text])

similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

print(f"Match Score: {similarity * 100:.2f}%")
