from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    return model.encode(text)

# def generate_embedding(text):
#     vectorizer = embedding_functions.DefaultEmbeddingFunction()
#     return vectorizer(text)

def cosine_sim(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))


