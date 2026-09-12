"""Lightweight BM25 scorer for lexical matching."""

import re
import math
from collections import Counter

# Common English stop words
STOP_WORDS = frozenset({
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'both', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'and', 'but', 'or', 'if', 'while',
    'about', 'up', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
    'he', 'him', 'his', 'she', 'her', 'they', 'them', 'their', 'this', 'that',
    'these', 'those', 'what', 'which', 'who', 'whom',
})


def tokenize(text: str) -> list[str]:
    """Tokenize text: lowercase, split on non-word chars, remove stop words."""
    tokens = re.findall(r'\w+', text.lower())
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


class BM25:
    """BM25 scorer over a corpus of documents."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[list[str]] = []
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.tf: list[Counter] = []

    def index(self, documents: list[dict]):
        """Index documents. Each doc: {id, text}."""
        self.corpus = []
        self.doc_ids = []
        self.doc_texts = []
        self.doc_len = []
        self.tf = []

        for doc in documents:
            tokens = tokenize(doc['text'])
            self.corpus.append(tokens)
            self.doc_ids.append(doc['id'])
            self.doc_texts.append(doc['text'])
            self.doc_len.append(len(tokens))
            self.tf.append(Counter(tokens))

        n = len(self.corpus)
        if n == 0:
            return

        self.avgdl = sum(self.doc_len) / n

        # Compute IDF
        df: dict[str, int] = {}
        for tokens in self.corpus:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str) -> list[tuple[str, str, float]]:
        """Score all documents against a query.
        Returns [(doc_id, doc_text, score)] sorted descending."""
        query_tokens = tokenize(query)
        if not query_tokens or not self.corpus:
            return [(did, dtxt, 0.0) for did, dtxt in zip(self.doc_ids, self.doc_texts)]

        scores = []
        for i, (doc_tf, dl) in enumerate(zip(self.tf, self.doc_len)):
            s = 0.0
            for term in query_tokens:
                if term not in self.idf:
                    continue
                tf_val = doc_tf.get(term, 0)
                idf_val = self.idf[term]
                numerator = tf_val * (self.k1 + 1)
                denominator = tf_val + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                s += idf_val * numerator / denominator
            scores.append((self.doc_ids[i], self.doc_texts[i], s))

        scores.sort(key=lambda x: x[2], reverse=True)
        return scores
