#!/usr/bin/env python3

import argparse

from lib.preprocessing import Preprocessing
from lib.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search cmd parser
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # Build cmd parser
    build_parser = subparsers.add_parser("build", help="Builts the Inverted Index for faster lookups")

    # tf (term frequency) cmd parser
    tf_parser = subparsers.add_parser("tf", help="List term frequency of a keyword in a given document")
    tf_parser.add_argument("doc_id", type=int, help="Id of the document to search")
    tf_parser.add_argument("term", type=str, help="Search the occurrence (freq.) of term in document object of 'doc_id'")

    args = parser.parse_args()

    match args.command:
        case "search":
            # Search (updating user search has begin)
            print(f"Searching for: {args.query}")

            # Loads the Cache file
            #index_cache, docmap_cache, tf_cache = InvertedIndex().load('index.pkl','docmap.pkl', 'term_frequencies.pkl')
            index_cache = InvertedIndex().load('index.pkl')
            docmap_cache = InvertedIndex().load('docmap.pkl')

            tokens = Preprocessing(args.query).stemming()   # query text is processed {refer cli/preprocessing.py}
            tokens = list(set(tokens))  # removes duplicate tokens

            indexes = []    # contains index of movies, which matched the search
            for token in tokens:
                try:
                    indexes.extend(index_cache[token])
                except KeyError:
                    continue

            indexes = list(set(indexes))     # Remove duplicate indexes  
            indexes.sort()    # Sort the list

            # Output to the User
            i = 1
            for index in indexes:
                if i > 5:
                   break

                movie_object = docmap_cache[index]
                print(f"{i}. {movie_object['title']}\n")
                i += 1

        case "build":
            inverted_index = InvertedIndex()
            
            inverted_index.build()  # builds the cache
            inverted_index.save()   # saves the cache

        case "tf":
            frequency = InvertedIndex().get_tf(args.doc_id, args.term)

            print(frequency)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()