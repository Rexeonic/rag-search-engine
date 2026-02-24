import math

from sentence_transformers import SentenceTransformer

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

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

#####################################################################################################################
#                                                                                                                   #
#                                                                                                                   #
#####################################################################################################################
def verify_model():

    sem_model = SemanticSearch().model

    print(f"Model loaded: {sem_model}")

    print(f"Max sequence length: {sem_model.max_seq_length}")

def add_vectors(vec1, vec2):
        
    if len(vec1) != len(vec2):
        raise ValueError("Vector size must be same")
    
    sum_vec = [None] * len(vec1)
    for i in range(0, len(vec1)):
        sum_vec[i] = vec1[i] + vec2[i]

    #print(sum_vec)
    return sum_vec

def subtract_vectors(vec1, vec2):

    if len(vec1) != len(vec2):
        raise ValueError("Vector size must be same")
    
    sub_vec = [None] * len(vec1)
    for i in range(0, len(vec1)):
        sub_vec[i] = vec1[i] - vec2[i]

    #print(sub_vec)
    return sub_vec

def dot(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vector size must be same")
    
    dot = 0
    for i in range(0, len(vec1)):
        dot += vec1[i] * vec2[i]

    #print(dot_vec)
    return dot

def euclidean_norm(vec):
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

def cosine_similarity(vec1, vec2):

        if len(vec1) != len(vec2):
            raise ValueError('vectors must be of equal size')    

        # cosine_similarity = dot_product(A, B) / (magnitude(A)*magnitude(B))
        magnitude_a = euclidean_norm(vec1)
        magnitude_b = euclidean_norm(vec2)

        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        
        cosine_similarity = dot(vec1, vec2) / (magnitude_a * magnitude_b)

        return cosine_similarity