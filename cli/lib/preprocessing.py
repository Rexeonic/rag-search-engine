"""
    preprocessing.Preprocessing

        This class processes text for better searching.
        Refer {1.0.1} cli/notes.md

        __init__ 
            :params text: (public) str

            : Description
            It also stages the Case Insensitivity part by converting
            the text into lowercase

        remove_punctuation
            removes punctuation like !,$,`,~  present in the text
            Warning: might not remove hyphen (-)    {manually removed from tokens}

            :params translator: (private)
            :params punctuated_text: (private)

        tokenization
            'remove_punctuation' method is called and result is stored
                in 'punctuated_text'.

            'tokens' are created from 'punctuated_text' & is returned

            :params punctuated_text: (private) str
            :params tokens: (private) List
                
        stop_words
            'tokens' are created by calling tokenization method

                Low-value tokens are removed from 'tokens' list and
                stored into 'stop_words' list.

            :params tokens: (private) List
            :params stop_words: (private) List

            
                
        stemming
            'high_value_tokens' are created by calling the method 'stop_words'

            Porter algorithm is used to stem words to their base form using
            (PorterStemmer().stem() i.e stemmer.stem()) and stored into 
            'stemmed_tokens'

            :params stemmer: (private) Instance of PorterStemmer() class
            :params high_value_tokens: (private) List
            :params stemmed_tokens: (private) List

"""
from pathlib import Path    # common imports of classes

# class specific imports are above the classes
import string
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
        filepath = Path(__file__).resolve().parent.parent.parent/'data'/'stopwords.txt'    # for multi-OS support
        #print(filepath)
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

        stemmed_tokens = [ stemmer.stem(token) for token in high_value_tokens ]

        return stemmed_tokens

import json # class specific imports are above the classes

class GetData:
    """ Searches the /data for file and returns the 
        required data
    """
    def __init__(self, filename):
        self.filename = filename
        self.filepath = Path(__file__).resolve().parents[2]/'data'/self.filename

    def get_file(self):
        """ searches the file in /data directory 
            return file descriptor (in read mode) - if file is found
            return None - if file is not found
        """
        try:
            with open(self.filepath, 'r') as f:
                return f
        except FileNotFoundError:
                return None
        
    def get_file_data_json(self):
        """ uses json.load to load the json file 
            return None - if file not found
        """
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
                return None
    
    def get_file_data_txt(self):
        """ uses .read() method to read the file """
        pass


#file = GetData('movies.json').get_file()
#print(type(file))      