#!/usr/bin/env python3

import argparse

from lib.semantic_search import (verify_model, 
                                 embed_text,
                                 verify_embeddings,
                                 embed_query_text,
                                 search,
                                 SemanticSearch)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="verifies all-MiniLM-L6-v2 is properly loaded")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", 
                                                     help="verifies embeddings are present, if no, creates them")
    

    embed_text_parser = subparsers.add_parser("embed_text", help="generate embeddings for a given text")
    embed_text_parser.add_argument("text", type=str, help="text from which embedding will be generated")

    embed_query_parser = subparsers.add_parser("embedquery", help="generate embeddings for user provided query")
    embed_query_parser.add_argument("query", type=str, help="user query to be encoded into embedding")

    search_parser = subparsers.add_parser("search", help="semantic search for relevant results")
    search_parser.add_argument("query", type=str, help="user query on which search is performed")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="user query on which search is performed")

    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "verify_embeddings":
            verify_embeddings()

        case "embed_text":
            embed_text(args.text)

        case "embedquery":
            embed_query_text(args.query)

        case "search":
            result = search(args.query, args.limit)

            underline = '\033[4m'
            end = '\033[0m'
            for i, res in enumerate(result):
                print(f"{i+1}. {underline}{res['title']}{end} (score: {res['score']})\n\t {res['description']}\n\n")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()