import re

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

urls = [
    "https://en.wikipedia.org/wiki/Mahatma_Gandhi",
    "https://en.wikipedia.org/wiki/Sarojini_Naidu",
    "https://en.wikipedia.org/wiki/Virat_Kohli",
    "https://en.wikipedia.org/wiki/Lionel_Messi",
    "https://en.wikipedia.org/wiki/Narendra_Modi",
    "https://en.wikipedia.org/wiki/Ilya_Sutskever",
]
headers = {
    "User-Agent": "SmrutiSemanticSearchBot/1.0 (contact: smruti45ranjan@gmail.com)"
}

responses = []
for url in urls:
    responses.append(requests.get(url, headers=headers))

htmls = []
for response in responses:
    htmls.append(response.text)

all_text = []
for html in htmls:
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = soup.find_all("p")
    raw_text = " ".join([p.get_text() for p in paragraphs])
    all_text.append(raw_text)

combined_text = " ".join(all_text)
text = re.sub(r"\[\d+\]", "", combined_text)
text = text.strip()

documents = [s.strip() for s in text.split("\n") if s.strip()]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer("all-MiniLM-L6-v2")
model.to(device)

doc_embeddings = model.encode(documents)


def semantic_search(query, top_k=5):
    query_embedding = model.encode([query])
    similarity = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarity)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({"document": documents[idx], "score": float(similarity[idx])})

    return results


result = semantic_search("When was gandhi died??", 10)
for r in result:
    print(r["document"], "score: ", r["score"])
