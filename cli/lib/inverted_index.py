from pathlib import Path
import pickle
import bisect

from .preprocessing import Preprocessing, GetData

class InvertedIndex:
    def __init__(self):
        self.index = {} # mapping tokens (strings) to sets of document IDs (integers)
        self.docmap = {} # mapping document IDs to their full document objects

        movies_data = GetData('movies.json').get_file_data_json()
        self.movies = movies_data['movies']     # is a list[dict{'id':, 'title':, 'description':}]

    def __add_document(self, doc_id, text):
        """
        Docstring for __add_document: 
            * Tokenizes the text
            * for each token, call get_document() and 
              stores the result in index
            * also insert doc_id {if not only present}
 
        :param doc_id: 'id' of movie which the text belong to
        :param text: contain movie['title'] and movie['description'] as
                     a single string
        """
        tokens = Preprocessing(text).stop_words()   # removed non-relevant words
        
        for token in tokens:
            self.index[token] = self.get_document(token)
           
            if doc_id not in self.index[token]:
                 # insert doc_id at right index
                bisect.insort(self.index[token], doc_id)

    def get_document(self, term) -> list[int]:
        """
        :param term: usually a single token

        get the set of document IDs for a given token, 
        and return them as a list, sorted in ascending order
        """
        # get document id's for a given token
        doc_id  = [ movie['id'] for movie in self.movies if term.lower() in movie['title'].lower()]
        
        # sort doc_id in ascending order 
        if len(doc_id) > 1:
            doc_id.sort()

        return doc_id

    def build(self):
        """
        Builds the Inverted Index for faster 
        lookups.

        Builds index and docmap dictionaries
        then calls the save() method
        """
        for movie in self.movies:
            # create the docmap dictionary
            self.docmap[movie['id']] = movie

            input_text = f"{movie['title']} {movie['description']}"
            # create the index dictionary
            self.__add_document(movie['id'], input_text)

        # Build Complete , now Save
        self.save()

            

    def save(self):
        """
        Save index, docmap dictionaries &
        saves them 
            @ cache/index.pk1
            @ cache/docmap.pk1
        """
        file_path = Path(__file__).resolve().parents[2]/'cache'
        
        # exist_ok=True (if dir already exist, don't raise error)
        # parents=True (create any necessary parent directories that don't exist)
        file_path.mkdir(exist_ok=True)

        with open(file_path/'index.pk1', 'wb') as index_file:
            pickle.dump(self.index, index_file)

        with open(file_path/'docmap.pk1', 'wb') as docmap_file:
            pickle.dump(self.docmap, docmap_file)

#example = InvertedIndex().build()
