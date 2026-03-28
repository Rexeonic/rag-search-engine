# Standard Library
from pathlib import Path
from operator import itemgetter

# Internal Dependency
from lib.preprocessing import GetData

# External Dependencies
from PIL import Image # pillow library (but imports as PIL)
from sentence_transformers import SentenceTransformer
import numpy as np

movies_data = GetData(Path(__file__).resolve().parents[2]/'data'/'movies.json').get_file_data_json()
documents = movies_data['movies']

dir_path = Path(__file__).resolve().parents[2]

def verify_image_embedding(image_path):
    print(dir_path/image_path)
    
    embedding = MultimodalSearch(documents, 'clip-ViT-B-32').embed_image(dir_path/image_path)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):

    return MultimodalSearch(documents).search_with_image(dir_path/image_path)


class MultimodalSearch:

    def __init__(self, documents, model_name='clip-ViT-B-32'):
        
        self.model = SentenceTransformer(model_name)
        self.documents = documents

        self.texts = [] # list["title: description"]
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")

        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path):
        
        print(image_path)
        image = Image.open(image_path)

        img_embedding = self.model.encode([image], show_progress_bar=True)

        return img_embedding[0] #

    def search_with_image(self, image_path):
        
        img_embedding = self.embed_image(image_path)

        result = []
        for doc, text_embedding in zip(self.documents, self.text_embeddings):
            similarity_score = self._cosine_similarity(img_embedding, text_embedding)

            result.append({
                'id': doc['id'],
                'title': doc['title'],
                'description': doc['description'][:200],
                'similarity_score': similarity_score 
            })

        result.sort(key=itemgetter('similarity_score'), reverse=True)

        return result[:5]   # returns Top 5 result
        

    def _cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)