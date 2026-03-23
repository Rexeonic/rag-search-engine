from lib.preprocessing import GetData
from lib.hybrid_search import HybridSearch
from parameters import RRF_K

class ManualEvaluation:

    def __init__(self):
        movies_dataset = GetData('movies.json').get_file_data_json()
        self.documents = movies_dataset['movies']

    def evaluate(self, test_cases: list[dict], limit: int):
        """
        :params test_cases: test cases in form ["query": str, 
                                                "relevant_docs": list[str]
                                               ]
        :params limit: Top-k results that are to be returned
        """
        search = HybridSearch(self.documents)

        test_result = []
        for test in test_cases:
                
            rrf_result = search.rrf_search(test['query'], RRF_K, limit)

            relevant_retrieved = []
            retrieved = []
            for result in rrf_result:

                retrieved.append(result['title'])
                if result['title'] in test['relevant_docs']:
                    relevant_retrieved.append(result['title'])

            precision_score = self.precision_value(relevant_retrieved, retrieved)
            recall_score = self.recall_value(relevant_retrieved, test['relevant_docs'])
            f1_score = self.f1_score(precision_score, recall_score)
           
            test_result.append({
                    "query": test['query'],
                    "precision": precision_score,
                    "recall": recall_score,
                    "f1": f1_score,
                    "retrived": f"{retrieved}".strip("[]"),
                    "relevant": f"{relevant_retrieved}".strip("[]")
                })
            

        print(f"k={limit}\n\n")
        for result in test_result:
            print(f"- Query: {result['query']}\n")
            print(f"\t- Precision@{limit}: {result['precision']}\n")
            print(f"\t- Recall@{limit}: {result['recall']}\n")
            print(f"\t- F1 Score: {result['f1']}\n")
            print(f"\t- Retrieved: {result['retrived']}\n")
            print(f"\t- Relevant: {result['relevant']}\n\n\n")

        #return test_result
    
    def precision_value(self, relevant_retrieved, total_retrieved):
        """
            Measures how many results are actually relevant in the 
            Search System
        """
        # precision = relevant_retrieved / total_retrieved
        return len(relevant_retrieved) / len(total_retrieved)


    def recall_value(self, relevant_retrieved, total_relevant):
        """
            Recall measures completeness. It tells you what percentage of
            all relevant documents you actually retrieved
        """
        # recall = relevant_retrieved / total_relevant
        return len(relevant_retrieved) / len(total_relevant)
    
    def f1_score(self, precision, recall):
        """
         balances precision and recall when both are equally important.

         F1 score is the harmonic mean of precision and recall. It gives you 
         one number that represents the overall performance of your search system.
        """

        # f1 = 2 * (precision * recall) / (precision + recall)
        return  2 * (precision * recall) / (precision + recall)