# Standard Libraries
import os
from operator import itemgetter
import pickle

# Internal Dependencies
from lib.inverted_index import (InvertedIndex,
                                file_path)
from lib.semantic_search import ChunkedSemanticSearch
 

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        #if not os.path.exists(self.idx.index_path):
        if not os.path.exists(file_path/'index.pkl'):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        #self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def hybrid_score(self, bm25_score, semantic_score, alpha=0.5):
        return (alpha * bm25_score) + ((1 - alpha) * semantic_score)
    
    def rrf_score(self, rank, k=60):
        if rank == 0:
            return rank
        
        return 1 / (k + rank)

    def weighted_search(self, query, alpha, limit):
        """
            *abbr
                css - chunked semantic search
        """
        # dict{id: score}
        bm25_result = self._bm25_search(query, limit*500)
        # list[dict{id:, title:, document:, score:, metadata:}]
        css_result = self.semantic_search.search_chunks(query, limit*500)

        norm_scores = self._normalize_score(bm25_result, css_result)

        norm_bm25_result = { res: norm_scores[idx] for idx, res in enumerate(bm25_result.keys()) }
        norm_css_result = { res['id']: norm_scores[idx + len(bm25_result)] for idx, res in enumerate(css_result) }
        
        # Union (|) to include results from EITHER search type
        common_id = set(norm_bm25_result.keys()) | set(norm_css_result.keys())

        try:
            with open(file_path/'docmap.pkl', 'rb') as f:
                doc_map = pickle.load(f)
        except FileNotFoundError:
            raise Exception(f"Cached file: docmap.pkl don't exist")
        
        hybrid_res = []
        for doc_id in common_id:
             # Default to 0.0 if the doc is missing from one result set
            k_score = norm_bm25_result.get(doc_id, 0.0)
            s_score = norm_css_result.get(doc_id, 0.0)
    
            document = doc_map[doc_id]

            hybrid_res.append({'doc_id': doc_id,
                               'title': document['title'],
                               'document': document['description'][:100],
                               'keyword_score': k_score,
                               'semantic_score': s_score,
                               'hybrid_score': self.hybrid_score(k_score, s_score, alpha)
                              })
            
        
        hybrid_res.sort(key=itemgetter('hybrid_score'), reverse=True)

        return hybrid_res[:limit]

    def _normalize_score(self, bm25_result, css_result):

        bm25_scores = [ score for score in bm25_result.values() ] # list of bm25 scores
        css_scores = [ res['score'] for res in css_result ]

        combined_scores = [*bm25_scores, *css_scores]
        
        # Normalized scores list returned
        return max_min_normalization(combined_scores)

    def rrf_search(self, query, k, limit):
        # dict{id: score}
        bm25_result = self._bm25_search(query, limit*500)
        # list[dict{id:, title:, document:, score:, metadata:}]
        css_result = self.semantic_search.search_chunks(query, limit*500)
    
        bm25_rank = { doc_id: idx+1 for idx, doc_id in enumerate(bm25_result.keys()) }
        css_rank = { res['id']: idx+1 for idx, res in enumerate(css_result) }

        # Union (|) to include results from EITHER search type
        common_id = set(bm25_rank.keys()) | set(css_rank.keys())

        try:
            with open(file_path/'docmap.pkl', 'rb') as f:
                doc_map = pickle.load(f)
        except FileNotFoundError:
            raise Exception(f"Cached file: docmap.pkl don't exist")
        
        rrf_res = []
        for doc_id in common_id:
            # Default to 0.0 if the doc is missing from one result set
            key_rank = bm25_rank.get(doc_id, 0)
            sem_rank = css_rank.get(doc_id, 0)
            document = doc_map[doc_id]

            # Reciprocal Rank Fusion score
            rrf_score = self.rrf_score(key_rank, k) + self.rrf_score(sem_rank, k)

            rrf_res.append({'doc_id': doc_id,
                            'title': document['title'],
                            'document': document['description'][:100],
                            'keyword_rank': key_rank,
                            'semantic_rank': sem_rank,
                            'rrf_score': rrf_score
                           })
            
        rrf_res.sort(key=itemgetter('rrf_score'), reverse=True)

        return rrf_res[:limit]


def max_min_normalization(scores):
        if not scores:  return []
        
        max_score = max(scores)
        min_score = min(scores)

        if max_score == min_score:
            return [1.0]*len(scores)
        
        norm_score = []
        for score in scores:
            norm_value = (score - min_score) / (max_score - min_score)

            norm_score.append(norm_value)

        return norm_score