from pathlib import Path
from math import log
from operator import itemgetter
from collections import defaultdict, Counter
import pickle

from lib.preprocessing import Preprocessing, GetData
from parameters import BM25_K1, BM25_B

file_path = Path(__file__).resolve().parents[2]/'cache'

class InvertedIndex:
    def __init__(self):
        # mapping tokens (strings) to sets of document IDs (integers)
        self.index = defaultdict(list)
        # mapping document IDs to their full document objects
        self.docmap = {}
        # mapping document IDs to Counter object i.e {doc_id: {token: frequency, ...},.... }
        self.term_frequencies = defaultdict(Counter)
        # maps document IDs to its Size (no. of tokens) i.e {doc_id: size,...}
        self.doc_lengths = {}

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
        tokens = Preprocessing(text).stemming()
        
        self.doc_lengths[doc_id] = len(tokens)
        # count frequency of term in a given document object
        self.term_frequencies[doc_id].update(tokens)
        for token in set(tokens):
            # add list elements in index
            # key=token is initilized, if not available
            self.index[token].append(doc_id)
    
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
        try:
            return term_frequency_cache[doc_id][token]
        except KeyError:
            return 0

    
    def get_idf(self, term):

        term = Preprocessing(term).stemming().pop()

        # Find total_doc, term_match_doc_count
        no_of_docs = len(self.load('term_frequencies.pkl'))
        term_match_doc_count = self._cal_df(term)

        # IDF = log ( N / df ) 
        #       where, 
        #           N -> total docs & 
        #           df -> no. of docs containing the term
        # +1 prevents division by zero when a term doesn't appear in any documents.
        idf = log( (no_of_docs + 1) / (term_match_doc_count + 1) )

        return idf
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)

        doc_length = self.load('doc_lengths.pkl')[doc_id]

        try:
            # length normalization factor,
            # length_norm = 1 - b + b * (doc_length / avg_doc_length)
            length_norm = 1 - b + b * (doc_length / self._get_avg_doc_length())

            # if b = 0 -> Normalization doesn't take place
            # if b = 1 -> Full Normalization effect takes place
        except ZeroDivisionError:
            length_norm = 0
        try:   
            # Saturated tf score
            bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        except ZeroDivisionError:
            return 0.0

        return bm25_tf
    
    def get_bm25_idf(self, term) -> float:
        """
        Docstring for get_bm25_idf
        
        
        :param term: term for which BM25 IDF score
                     is to be found
            Note -> should be a single token
        """
        # Find term_match_doc_count
        no_of_docs = len(self.load('term_frequencies.pkl'))
        term_match_doc_count = self._cal_df(term)

        # IDF = log( (N - df + 0.5)/(df + 0.5) + 1) 
        #       where, 
        #           N -> total docs 
        #           df -> no. of docs containing the term
        # +1 prevents division by zero when a term doesn't appear in any documents.
        bm25_idf = log( (no_of_docs - term_match_doc_count + 0.5) / (term_match_doc_count + 0.5) + 1 )

        return bm25_idf

    def bm25(self, doc_id, term):
        bm25_score = self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

        return bm25_score
    
    def bm25_search(self, query, limit):
        query = Preprocessing(query).stemming()

        size = len(self.load('docmap.pkl'))
        # maps document IDs to their total BM25 score
        scores = {}
 
        for doc_id in range(1, size + 1):
            total_bm25_score = 0
            for token in query:
                total_bm25_score += self.bm25(doc_id, token)

            scores[doc_id] = total_bm25_score
        
        scores.sort(key=itemgetter(1), reverse=True)

        # return top limit (default 5) results
        return dict(list(scores)[0:limit])

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
            #print(self.index)
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

        with open(file_path/'doc_lengths.pkl', 'wb') as doc_lengths_file:
            pickle.dump(self.doc_lengths, doc_lengths_file)

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
        
    # Private Helper Methods
    def _cal_df(self, term):
        """ Finds total_documents & term_match_doc_count """

        index_cache = self.load('index.pkl')

        # Finds MATCHing Document COUNT
        term_match_doc_count = len(index_cache[term])

        return term_match_doc_count
        
    def _get_avg_doc_length(self) -> float:
        """
        Docstring for _get_avg_doc_length
    
        Calculates and returns the average
        document length across all documents.

        :return avg_doc_length: avg. doc length of all 
                 document across dataset
        """
        doc_lengths_cache = self.load('doc_lengths.pkl')
        no_of_docs = len(self.load('docmap.pkl'))

        total_length_of_docs = 0
        for length in doc_lengths_cache.values():
            total_length_of_docs += length  # total size of all document included
        
        avg_doc_length = total_length_of_docs / no_of_docs
        #print(f"Avg. Doc Length: {avg_doc_length}")

        return avg_doc_length
