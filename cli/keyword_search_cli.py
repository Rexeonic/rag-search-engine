#!/usr/bin/env python3

import argparse
import json
from operator import itemgetter
from pathlib import Path

from preprocessing import Preprocessing

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # Search (updating user search has begin)
            print(f"Searching for: {args.query}")

            results = []     # Output of search

            dataset_path = Path(__file__).parent.parent / 'data' / 'movies.json'
            with open(dataset_path) as f:
                movies_data = json.load(f)      # is a dictionary

            # list of dictionaries of format [{ 'id': , 'title':, 'description':},...]
            movies = movies_data['movies']

            tokens = Preprocessing(args.query).stemming()   # query text is processed (refer cli/notes.md {1.0.1})
            for movie in movies:    # searching the dataset
                if any(token in movie['title'].lower() for token in tokens):
                    if len(results) == 5:   #limiting search upto 5 results
                        break
                    
                    results.append(movie)
            
            # sort the results-list (containing dict elements) using the key='id'
            results.sort(key=itemgetter('id'))

            # Output to the User
            i = 1
            for result in results:
                print(f"{i}. {result['title']}\n")
                i += 1
   
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()