#!/usr/bin/env python3

import argparse

from lib.preprocessing import Preprocessing
from lib.inverted_index import InvertedIndex
from parameters import BM25_K1, BM25_B


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

    # idf (inverse document frequency) cmd parser
    idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency: qualifies a term as rare, common, universal")
    idf_parser.add_argument("term", type=str, help="term for which IDF score is to be found")

    # tfidf (TF-IDF) cmd parser
    tf_parser = subparsers.add_parser("tfidf", help="List term frequency of a keyword in a given document")
    tf_parser.add_argument("doc_id", type=int, help="Id of the document to search")
    tf_parser.add_argument("term", type=str, help="Search the occurrence (freq.) of term in document object of 'doc_id'")

    # BM25 tf (term frequency) cmd parser
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    # BM25 idf (inverse document frequency)cmd parser
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="term for which BM25 IDF score is to be found")

    # BM25 Search
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="return x (default 5) number of results")

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

        case "idf":
            idf = InvertedIndex().get_idf(args.term)
           
            print(f"Inverse Document Frequency of {args.term}: {idf:.2f}")

       
        case "tfidf":
            # Calculate tf & idf
            tf = InvertedIndex().get_tf(args.doc_id, args.term)
            idf = InvertedIndex().get_idf(args.term)
            
            # Cal TF-IDF
            tf_idf = tf * idf

            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25tf":
            saturated_bm25_tf = InvertedIndex().get_bm25_tf(args.doc_id, 
                                                            args.term, 
                                                            args.k1, 
                                                            args.b)
            print(f"{saturated_bm25_tf:.2f}")

        case "bm25idf":
            bm25_idf = InvertedIndex().get_bm25_idf(args.term)

            print(f"BM25 IDF score of {args.term}: {bm25_idf:.2f}")

        case "bm25search":
            results = InvertedIndex().bm25_search(args.query, args.limit)

            docmap_cache = InvertedIndex().load('docmap.pkl')
            
            for id, score in results.items():
                print(f"{i}. ({id}) {docmap_cache[id]['title']} - Score: {score}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()