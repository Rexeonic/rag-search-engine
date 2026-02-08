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
        output = self.preprocessing.remove_punctaution()
        self.assertEqual(output, 'the wonderful bear grizzly')

    def test_tokenisation(self):
        output = self.preprocessing.tokenisation()
        self.assertEqual(output, ['the','wonderful', 'bear', 'grizzly'])
    
if __name__ == '__main__':
    unittest.main()