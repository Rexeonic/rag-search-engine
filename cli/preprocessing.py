"""
Docstring for preprocessing.punctaution

    This module is responsible in punctuation removal in text
    
    Refer (1.0.1)Text Processing in /cli/notes.md

    Basic working:

        "Hello, world!" -> "hello world"
        "sci-fi" -> "scifi"
"""
import string

class Preprocessing:
    def __init__(self, text):
        self.text = text

    def remove_punctaution(self) -> str:
        translator = str.maketrans('', '', string.punctuation)
        punctuated_text = self.text.translate(translator)

        return punctuated_text
        
    def tokenisation(self):
        punctuated_text = self.remove_punctaution()
        tokens = punctuated_text.split()

        return tokens
    
    def stop_words(self):
        tokens = self.tokenisation()
        words = []

        return words
    
    def stemming(self):
        pass

#text_1 = 'Boots the bear!'
#text_2 = 'The wonderful bear, Boots'
#punct_1 = Preprocessing(text_1).tokenisation()
#punct_2 = Preprocessing(text_2).tokenisation()

#print(punct_1)
#print(punct_2)