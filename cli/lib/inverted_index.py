from preprocessing import Preprocessing

class InvertedIndex:
    def __init__(self):
        self.index = {} # mapping tokens (strings) to sets of document IDs (integers)
        self.docmap = {} # mapping document IDs to their full document objects

    def __add_document(self, doc_id, text):
        tokens = Preprocessing(text).tokenization()
        
        for token in tokens:
            self.index[token].append(doc_id)

    def get_document(self, term):
        pass

    def build(self):
        pass

    def save(self):
        pass
