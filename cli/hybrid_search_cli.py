import argparse
from pathlib import Path

from lib.preprocessing import GetData
from lib.hybrid_search import (HybridSearch,
                               max_min_normalization)
from parameters import (ALPHA,
                        RRF_K)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="normalizes the BM25 and Semantic search scores")
    normalize_parser.add_argument("scores", type=float, nargs='+', help="list of scores to be normalized")

    weighted_search_parser = subparsers.add_parser("weighted_search", help="Hybrid searc with weighted average combination")
    weighted_search_parser.add_argument("query", type=str, help="query to be searched")
    weighted_search_parser.add_argument("--alpha", type=float, default=ALPHA, help="alpha parameter controls the weighting")
    weighted_search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="limit of results (to be returned)")

    rrf_search_parser = subparsers.add_parser("rrf_search", help="Reciprocal Rank Fusion Search")
    rrf_search_parser.add_argument("query", type=str, help="query to be searched")
    rrf_search_parser.add_argument("--k", type=int, default=RRF_K, help="weight given to higher-ranked results vs. lower-ranked results")
    rrf_search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="limit of results (to be returned)")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            score = args.scores # list of scores (float values)
            
            norm_score = max_min_normalization(score)
            for val in norm_score:
                print(f"* {val:.4f}")

        case "weighted_search":
            movies_data = GetData(Path(__file__).resolve().parents[1]/'data'/'movies.json').get_file_data_json()
            documents = movies_data['movies']

            results = HybridSearch(documents).weighted_search(args.query, args.alpha, args.limit)
        
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']}")
                print(f"Hybrid Score: {result['hybrid_score']}")
                print(f"BM25: {result['keyword_score']}, Semantic: {result['semantic_score']}")
                print(f"{result['document']}...\n\n")

        case "rrf_search":
            movies_data = GetData(Path(__file__).resolve().parents[1]/'data'/'movies.json').get_file_data_json()
            documents = movies_data['movies']

            results = HybridSearch(documents).rrf_search(args.query, args.k, args.limit)
        
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']}")
                print(f"RRF Score: {result['rrf_score']}")
                print(f"BM25 Rank: {result['keyword_rank']}, Semantic Rank: {result['semantic_rank']}")
                print(f"{result['document']}...\n\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()