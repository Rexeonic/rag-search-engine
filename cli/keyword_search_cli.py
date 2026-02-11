#!/usr/bin/env python3

import argparse
from operator import itemgetter
from pathlib import Path
import pickle

from lib.preprocessing import Preprocessing, GetData
from lib.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build Inverted Index for faster lookups")
    # build_parser.add_argument("", type=None, help="Build query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # Search (updating user search has begin)
            print(f"Searching for: {args.query}")

            results = []     # Output of search

            movies_data = GetData('movies.json').get_file_data_json()   
            movies = movies_data['movies']  # is a list[dict{'id':, 'title':, 'description':}]

            tokens = Preprocessing(args.query).stemming()   # query text is processed {refer cli/preprocessing.py}
            for movie in movies:    # searching the dataset
                if any(token in movie['title'].lower() for token in tokens):
                    if len(results) == 5:   #limiting search upto 5 results
                        break
                    
                    results.append(movie)
            
            # sort results ( list[dict] ) using the key='id'
            results.sort(key=itemgetter('id'))

            # Output to the User
            i = 1
            for result in results:
                print(f"{i}. {result['title']}\n")
                i += 1

        case "build":
            index_builder = InvertedIndex().build()

            docs_path = Path(__file__).resolve().parents[1]/'cache'/'index.pk1'
            with open(docs_path, 'rb') as index_file:
                docs = pickle.load(index_file)

                print(f"First document for token 'merida' = {docs['merida']}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()