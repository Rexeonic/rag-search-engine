"""
    preprocessing.Preprocessing

        This class processes text for better searching.
        Refer {1.0.1} cli/notes.md

        __init__ 
            : params 
                text (public)

            : Description
            It also stages the Case Insensitivity part by converting
            the text into lowercase

        remove_punctuation 
            : params 
                translator (private)
                punctuated_text (private)

            : Description
            removes punctuation like !,$,`,~  present in the text

            Warning: might not remove hyphen (-)    {manually removed from tokens}

        tokenization
            : params  

"""
import string
from pathlib import Path
from nltk.stem import PorterStemmer

class Preprocessing:
    def __init__(self, text):
        self.text = text.lower()    # Case Insensitivity: make the text lowercase

    def remove_punctuation(self) -> str:
        translator = str.maketrans('', '', string.punctuation)
        punctuated_text = self.text.translate(translator)

        return punctuated_text
        
    def tokenization(self):
        """ Implements word-based tokenization for keyword search """
        punctuated_text = self.remove_punctuation()
        tokens = punctuated_text.split()

        return tokens
    
    def stop_words(self):
        tokens = self.tokenization()
        stop_words = []     # words which don't have much sematic meaning

        #filepath = f'{Path(__file__).resolve().parent.parent}/data/stopwords.txt'   # construct file path
        filepath = Path(__file__).resolve().parent.parent / 'data' / 'stopwords.txt'    # for multi-OS support
        try:
            with open(filepath) as f:
                for line in f:
                    stop_words.append(line.strip())

        except FileNotFoundError:
            print("File: stopwords.txt not found")

        for token in tokens:    # takes individual token from query
            if token in stop_words:    # checks if the token is a stop_word (low-value token)
                tokens.remove(token)    # Low-value tokens are removed
     
        return tokens
    
    def stemming(self):
        """
            Reduce tokens to their root form. 
            This helps match different variations of the same word
        """
        stemmer = PorterStemmer()
        high_value_tokens = self.stop_words()

        stemmed_tokens = [ stemmer.stem(token) for token in high_value_tokens]

        return stemmed_tokens