from pathlib import Path
from math import log
import pickle
import bisect

from lib.preprocessing import Preprocessing, GetData

file_path = Path(__file__).resolve().parents[2]/'cache'

class InvertedIndex:
    def __init__(self):
        # mapping tokens (strings) to sets of document IDs (integers)
        self.index = {}
        # mapping document IDs to their full document objects
        self.docmap = {}
        # mapping document IDs to Counter object i.e {doc_id: {token: frequency, ...},.... }
        self.term_frequencies = {}

        # Getting Data (later we'll use vector databases)
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
        #tokens = text.split()  # Use THIS: for cache that includes all words (even Low-Value tokens)
        tokens = Preprocessing(text).stemming()   # (for only High-Value tokens) !!! Has its own downside !!!
        for token in tokens:
            # add list elements in index
            # if key=token is initilized, if not available
            if token not in self.index: # so get_document() doesn't run multiple times on same tokens
                self.index.setdefault(token,[]).extend(self.get_document(token))

            if doc_id not in self.index[token]:
                # insert doc_id at right index (correct position acc. to ascending order)
                bisect.insort(self.index[token], doc_id)

            # count frequency of term in a given document object
            self.term_frequencies[doc_id][token] = self.term_frequencies.setdefault(doc_id, {}).setdefault(token, 0) + 1

    def get_document(self, term) -> list[int]:
        """
        :param term: usually a single token

        get the set of document IDs for a given token, 
        and return them as a list, sorted in ascending order
        """
        # get document id's for a given token
        doc_id_list  = [ movie['id'] for movie in self.movies if term.lower() in movie['title'].lower()]

        # sort doc_id in ascending order 
        # It will be a sorted list as parsed data is sorted (no need to sort)

        return doc_id_list
    
    def get_tf(self, doc_id, term):
        term = Preprocessing(term).stemming()

        if len(term) > 1:
            raise Exception("InvertedIndex.get_tf(): More than 1 token")
        else:
            try:
                token = term.pop()
            except IndexError:  # if term is empty (cann't pop empty list)
                return 0
        
        term_frequency_cache = self.load('term_frequencies.pkl')

        # if term exist returns counter
        return term_frequency_cache[doc_id][token]
    
    def get_bm25_tf(self, doc_id, term, k1):
        tf = self.get_tf(doc_id, term)

        # Saturated tf score
        bm25_tf = (tf * (k1 + 1)) / (tf + k1)

        return bm25_tf
    
    def get_idf(self, term):

        # Find total_doc, term_match_doc_count
        total_doc, term_match_doc_count = GETQUANTITIES().cal_N_and_df(term)

        # IDF = log ( N / df ) 
        #       where, 
        #           N -> total docs & 
        #           df -> no. of docs containing the term
        # +1 prevents division by zero when a term doesn't appear in any documents.
        idf = log( (total_doc + 1) / (term_match_doc_count + 1) )

        return idf
    
    def get_bm25_idf(self, term) -> float:
        """
        Docstring for get_bm25_idf
        
        
        :param term: term for which BM25 IDF score
                     is to be found
            Note -> should be a single token
        """
        # Find total_doc, term_match_doc_count
        total_doc, term_match_doc_count = GETQUANTITIES().cal_N_and_df(term)

        # IDF = log( (N - df + 0.5)/(df + 0.5) + 1) 
        #       where, 
        #           N -> total docs 
        #           df -> no. of docs containing the term
        # +1 prevents division by zero when a term doesn't appear in any documents.
        bm25_idf = log( (total_doc - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1 )

        return bm25_idf


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

        # Build Complete

    def save(self):
        """
        Save index, docmap, term_frequency dictionaries &
        saves them 
            @ cache/index.pkl
            @ cache/docmap.pkl
            @ cache/term_frequencies.pkl
        """   
        # exist_ok=True (if dir already exist, don't raise error)
        # parents=True (create any necessary parent directories that don't exist)
        file_path.mkdir(exist_ok=True, parents=False)

        with open(file_path/'index.pkl', 'wb') as index_file:
            pickle.dump(self.index, index_file)

        with open(file_path/'docmap.pkl', 'wb') as docmap_file:
            pickle.dump(self.docmap, docmap_file)

        with open(file_path/'term_frequencies.pkl', 'wb') as tf_file:
            pickle.dump(self.term_frequencies, tf_file)

    def load(self, filename):
        """
        Docstring for load

            loads() the cached file for search using 
            pickle.load() in read mode
        
        :param filenames: names of the cached file 
        """
        try:
            with open(file_path/filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            raise Exception(f"Cached file: {filename} don't exist")
        

class GETQUANTITIES:
    """
    Docstring for GETQUANTITIES

    Calculate quantities used in IDF formula
    """
    def __init__(self):
        self.index_cache = InvertedIndex().load('index.pkl')
        self.docmap_cache = InvertedIndex().load('docmap.pkl')

    def cal_N_and_df(self, term):
        """ Finds total_documents & term_match_doc_count """
        # Finds Total Documents (for this prototype its 5000)
        total_doc = len(self.docmap_cache) 
        # Finds Total MATCHing Document COUNT
        try:
            term_match_doc_count = len(self.index_cache[term])
        except KeyError:
            term = Preprocessing(term).stemming().pop()  # bcs it returns a list (so, pop() is used)
            term_match_doc_count = len(self.index_cache[term])

        return total_doc, term_match_doc_count