#!/usr/bin/env python3

import argparse
import json
from operator import itemgetter

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

            with open('data/movies.json') as f:
                movies_data = json.load(f)      # is a dictionary

            # list of dictionaries of format { 'id': , 'title':, 'description':}
            movies = movies_data['movies']

            tokens = Preprocessing(args.query.lower()).tokenisation()   # query text is converted into tokens
            for movie in movies:
                for token in tokens:
                    if token in movie['title'].lower():
                        if len(results) == 5:
                            break
                    
                        results.append(movie)

            # sort the results-list (containing dict) using the key='id'
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