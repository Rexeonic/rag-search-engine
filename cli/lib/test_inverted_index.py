import unittest

from inverted_index import InvertedIndex


class TestInvertedIndex(unittest.TestCase):
    
    def setUp(self):
        self.inverted_index = InvertedIndex()

    def test_bm25_idf(self, term) -> float:
        bm25_idf = self.inverted_index.get_bm25_idf(term)

        return bm25_idf

if __name__ == '__main__':
    unittest.main()