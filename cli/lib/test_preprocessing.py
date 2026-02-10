import unittest

from preprocessing import Preprocessing

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        text = 'The wonderful bear, Grizzly'
        self.preprocessing = Preprocessing(text)

    def test_remove_punctuation(self):
        """
        Docstring for test_remove_punctuation
        
        :param self
        :Description 
            removes punctuation. Might fail on hyphens(-)
        """    
        output = self.preprocessing.remove_punctuation()
        self.assertEqual(output, 'the wonderful bear grizzly')

    def test_tokenization(self):
        output = self.preprocessing.tokenization()
        self.assertEqual(output, ['the','wonderful', 'bear', 'grizzly'])

    def test_stop_words(self):
        output = self.preprocessing.stop_words()
        self.assertEqual(output, ['wonderful', 'bear', 'grizzly'])

    def test_stemming(self):
        output = self.preprocessing.stemming()
        self.assertEqual(output, ['wonder', 'bear', 'grizzli'])

if __name__ == '__main__':
    unittest.main()