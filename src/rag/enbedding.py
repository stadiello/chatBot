from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

class CustomEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input).tolist()

def cosine_sim(a, b):
    return np.dot(a, b)/(norm(a)*norm(b))

# Instance globale de la fonction d'embedding
generate_embedding = CustomEmbeddingFunction()