"""Lightweight local retrieval using TF-IDF cosine similarity."""
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def split_chunks(text: str, words_per_chunk: int = 110):
    words = text.split()
    return [" ".join(words[i:i + words_per_chunk])
            for i in range(0, len(words), words_per_chunk)]

class PolicyRetriever:
    def __init__(self, documents_dir="data/policy"):
        self.documents_dir = Path(documents_dir)
        self.chunks, self.sources = [], []
        for path in sorted(self.documents_dir.glob("*.md")):
            for chunk in split_chunks(path.read_text(encoding="utf-8")):
                self.chunks.append(chunk)
                self.sources.append(path.name)
        if not self.chunks:
            raise ValueError(f"No .md policy documents found in {self.documents_dir}")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def search(self, query: str, top_k: int = 3):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        indices = scores.argsort()[::-1][:top_k]
        return [{"text": self.chunks[i], "source": self.sources[i], "score": float(scores[i])}
                for i in indices if scores[i] > 0]
