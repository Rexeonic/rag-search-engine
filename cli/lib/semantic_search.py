# Standard Libraries
from pathlib import Path
from operator import itemgetter

# External Dependencies
import numpy as np
from sentence_transformers import SentenceTransformer

# Internal Dependencies
from lib.preprocessing import GetData

embeddings_file_path = Path(__file__).resolve().parents[2]/'cache'/'movie_embeddings.npy'

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

    
def embed_query_text(query):
 
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(query.strip())

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")



def search(query, limit):

    movies_data = GetData('movies.json').get_file_data_json()
    documents = movies_data['movies']

    sem_search = SemanticSearch()
    sem_search.load_or_create_embeddings(documents)
    result = sem_search.search(query, limit)

    return result

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.embeddings = None
        self.documents = None
        self.document_map = {}


    def generate_embedding(self, text):
        
        if not text or not text.strip():
            raise ValueError("Text is either all spaces or empty")
        
        text = text.strip()

        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents):
        # documents is a list[dict], where dict is a movie object
        self.documents = documents
        self.document_map = {}

        movie_strings = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc

            movie_strings.append(f"{doc['title']}:{doc['description']}")

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        
        # save array to binary file
        np.save(embeddings_file_path, self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents

        for doc in self.documents:
            self.document_map[doc['id']] = doc

        if embeddings_file_path.exists():
            self.embeddings = np.load(embeddings_file_path, 'r')

            if len(self.embeddings) == len(self.documents):
                return self.embeddings
                
        else:
            self.embeddings = self.build_embeddings(documents)
            return self.embeddings
    
    def search(self, query, limit):
        if self.embeddings.all() == None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        similarity_score = []
        for doc, doc_embedding in zip(self.documents , self.embeddings):
            _similarity = self._cosine_similarity(doc_embedding, query_embedding)
            
            similarity_score.append( (_similarity, doc) )

        similarity_score.sort(key=itemgetter(0), reverse=True)
        #print(sem_search)
        res = []
        for i, value in enumerate(similarity_score):
            if i >= limit:
                break

            item = {'score': value[0], 
                    'title': value[1]['title'],
                    'description': value[1]['description'] }
            res.append(item)

        return res

    def _cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

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


