#!/usr/bin/env python3

import argparse
from pathlib import Path

from lib.preprocessing import GetData
from lib.semantic_search import (verify_model, 
                                 embed_text,
                                 verify_embeddings,
                                 embed_query_text,
                                 search,
                                 semantic_chunk,
                                 ChunkedSemanticSearch)

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
    
    chunk_parser = subparsers.add_parser("chunk", help="split long text into smaller pieces for embedding")
    chunk_parser.add_argument("text", type=str, help="text to chunk")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="how many words chunks share to preserve context")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="size for a single chunk")
    
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="semantic chunking for text")
    semantic_chunk_parser.add_argument("text", type=str, help="text to be chunked semantically")
    semantic_chunk_parser.add_argument("--max-chunk-size",type=int, nargs='?', default=4, help="max chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="text to be chunked semantically")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="generates embeddings for all the chunks")


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

        case "chunk":
            overlap = args.overlap
            chunk_size = args.chunk_size

            if overlap >= chunk_size:
                raise ValueError("overlap cann't be greater than chunk size")
            
            words = args.text.split()

            chunks = []
            for i in range(0, len(words), chunk_size-overlap):
                chunk = words[i:i+chunk_size]

                if len(chunk) <= overlap:
                    break

                chunks.append(" ".join(chunk))

            # printing for user
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i+1}. {chunk}")

        case "semantic_chunk":
            results = semantic_chunk(args.text, args.max_chunk_size, args.overlap)

            print(f"Semantically Chunking {len(args.text)} characters")
            for i, result in enumerate(results):
                print(f"{i+1}. {result}")

        case "embed_chunks":
            movies_data = GetData(Path(__file__).resolve().parents[1]/'data'/'movies.json').get_file_data_json()
            documents = movies_data['movies']

            embeddings = ChunkedSemanticSearch().load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()