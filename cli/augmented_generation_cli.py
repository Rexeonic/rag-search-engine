import argparse
from pathlib import Path

from lib.hybrid_search import HybridSearch
from lib.preprocessing import GetData
from lib.llm import LlmPrompt
from parameters import RRF_K

movies_data = GetData(Path(__file__).resolve().parents[1]/'data'/'movies.json').get_file_data_json()
documents = movies_data['movies']

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Generates Summary from retrieved results"
    )
    summarize_parser.add_argument("query", type=str, help="Query for movie search")
    summarize_parser.add_argument("--limit", type=int, nargs='?', default=5 ,help="Number of results to be returned")


    citation_parser = subparsers.add_parser(
        "citation", help="Generates Summary with citations"
    )
    citation_parser.add_argument("query", type=str, help="Query for movie search")
    citation_parser.add_argument("--limit", type=int, nargs='?', default=5 ,help="Number of results to be returned")
    
    question_parser = subparsers.add_parser(
        "question", help="Answers user's question"
    )
    question_parser.add_argument("query", type=str, help="User's question")
    question_parser.add_argument("--limit", type=int, nargs='?', default=5 ,help="Number of results to be returned")
    

    args = parser.parse_args()

    match args.command:
        case "rag":
            limit = 5
            # do RAG stuff here
            hybrid_search = HybridSearch(documents)

            # Search Result from Reciprocal Rank Fusion
            rrf_result = hybrid_search.rrf_search(args.query, RRF_K, 5*limit)
            # Augmentation of RRF Result to LLM
            llm_response = LlmPrompt('gemma-3-27b-it').llm_augmentation(args.query, rrf_result[:limit])
            
            ############### Printing Search Results to User ##############
            print("Search Results:")
            for movie in rrf_result[:limit]:
                print(f"- {movie['title']}")

            print("RAG Response:")
            print(llm_response)
            
        case "summarize":
            hybrid_search = HybridSearch(documents)

            # Search Result from Reciprocal Rank Fusion
            rrf_result = hybrid_search.rrf_search(args.query, RRF_K, args.limit*5)
            # Augmentation of RRF Result to LLM
            llm_response = LlmPrompt('gemma-3-27b-it').llm_summarization(args.query, rrf_result[:args.limit])


            ############### Printing Search Results to User ##############
            print("Search Results:")
            for movie in rrf_result[:args.limit]:
                print(f"- {movie['title']}")

            print("LLM Summary:")
            print(llm_response)        

        case "citation":
            hybrid_search = HybridSearch(documents)

            # Search Result from Reciprocal Rank Fusion
            rrf_result = hybrid_search.rrf_search(args.query, RRF_K, args.limit*5)
            # Augmentation of RRF Result to LLM
            llm_response = LlmPrompt('gemma-3-27b-it').llm_citation(args.query, rrf_result[:args.limit])


            ############### Printing Search Results to User ##############
            print("Search Results:")
            for movie in rrf_result[:args.limit]:
                print(f"- {movie['title']}")

            print("LLM Answer:")
            print(llm_response)

        case "question":
            hybrid_search = HybridSearch(documents)

            question = args.query
            # Search Result from Reciprocal Rank Fusion
            rrf_result = hybrid_search.rrf_search(question, RRF_K, args.limit*5)
            # Augmentation of RRF Result to LLM
            llm_response = LlmPrompt('gemma-3-27b-it').llm_qna(question, rrf_result[:args.limit])


            ############### Printing Search Results to User ##############
            print("Search Results:")
            for movie in rrf_result[:args.limit]:
                print(f"- {movie['title']}")

            print("Answer:")
            print(llm_response)
            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()