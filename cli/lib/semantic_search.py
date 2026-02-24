# Standard Libraries
from pathlib import Path

# External Dependencies
import numpy as np
from sentence_transformers import SentenceTransformer

# Internal Dependencies
from lib.preprocessing import GetData

file_path = Path(__file__).resolve().parents[2]/'cache'

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if text == None or len(text.strip()) == 0:
            raise ValueError("Text is either all spaces or empty")

        embedding = self.model.encode(list(text))   # expects a list

        return embedding[0]
    
    def build_embeddings(self, documents):
        # documents is a list[dict], where dict is a movie object
        self.documents = documents  

        movies_string = []
        for index, doc in enumerate(self.documents):
            self.document_map[index] = doc

            movie_string = f"{doc['title']}:{doc['description']}"
            movies_string.append(movie_string)

        self.embeddings = self.model.encode(movies_string, show_progress_bar=True)
        
        # save array to binary file
        np.save(file_path/'movie_embeddings.npy', self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for index, doc in enumerate(self.documents):
            self.document_map[index] = doc

        try:
        
            self.embeddings = np.load(file_path/'movie_embeddings.npy', 'r')

            if len(self.embeddings) == len(self.documents):
                return self.embeddings
                
        except  FileNotFoundError:
            self.embeddings = self.build_embeddings(documents)
            return self.embeddings
    
    def cosine_similarity(self, vec1, vec2):

        if len(vec1) != len(vec2):
            raise ValueError('vectors must be of equal size')    

        # cosine_similarity = dot_product(A, B) / (magnitude(A)*magnitude(B))
        magnitude_a = self._euclidean_norm(vec1)
        magnitude_b = self._euclidean_norm(vec2)

        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        
        cosine_similarity = self._dot_product(vec1, vec2) / (magnitude_a * magnitude_b)

        return cosine_similarity
    
    def _euclidean_norm(self, vec):
        """
        euclidean_norm just adds the squares of all the numbers in a vector, 
        then takes the square root of that sum. 
        
        This should be reminiscent of the Pythagorean theorem.
        """
        sum_of_sqaures = 0.0
        for i in range(0, len(vec)):
            sum_of_sqaures += vec[i]**2
            
        euclidean_norm = sum_of_sqaures**0.5
        return euclidean_norm

    def _dot_product(self, vec1, vec2):
        if len(vec1) != len(vec2):
            raise ValueError("Vector size must be same")
    
        dot = 0.0
        for i in range(0, len(vec1)):
            dot += vec1[i] * vec2[i]

        return dot

def verify_model():

    sem_model = SemanticSearch().model

    print(f"Model loaded: {sem_model}")

    print(f"Max sequence length: {sem_model.max_seq_length}")

def embed_text(text):

    sem_search = SemanticSearch()

    embedding = sem_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():

    sem_search = SemanticSearch()

    movies_data = GetData('movies.json').get_file_data_json()
    documents = movies_data['movies']

    embeddings = sem_search.load_or_create_embeddings(documents)

    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
