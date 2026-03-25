# Standard Libraries
import argparse
import json
from operator import itemgetter
from pathlib import Path
import time, datetime

# External Dependencies
from lib.llm import LlmPrompt


# Internal Dependencies
from lib.preprocessing import GetData
from lib.hybrid_search import (HybridSearch,
                               max_min_normalization,
                               logger)
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
    rrf_search_parser.add_argument("--enhance", type=str, nargs='?', choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, nargs='?', choices=["individual","batch","cross_encoder"], help="Re-ranking for RRF search")
    rrf_search_parser.add_argument("--evaluate", action='store_true', help="Evaluate Search results using LLM")
    rrf_search_parser.add_argument("--debug", action='store_true', help="Log data for debugging")

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

            ######################################################################################################
            ######################################################################################################
            if args.debug:                                                              #                        #
                now = datetime.datetime.now()   # logging timestamp                     #  Generating logs @     #
                                                                                        # rag-search-engine/logs #
                logger( f"\n\n[ {now.strftime('%Y-%m-%d %I:%M:%S %p')} ]\n" )           #                        #
                logger( f"\tQuery: {args.query}\n" )                                    #                        #
            ######################################################################################################


            ####################### LLMs before the search to help improve the QUERY ############################# 
            # if enhance is provided (enhance user's query using LLM)
            if args.enhance:
                logger( f"\tEnhance option: {args.enhance} ✅SELECTED\n" )
                if args.enhance == "spell":
                    response = LlmPrompt('gemma-3-27b-it').spell(args.query)
                    
                elif args.enhance == "rewrite":
                    response = LlmPrompt('gemma-3-27b-it').rewrite(args.query)

                elif args.enhance == "expand":
                    response = LlmPrompt('gemma-3-27b-it').expand(args.query)
             
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{response}'\n")
                
                args.query = response

                logger( f"\tEnhanced Query: {args.query}\n" )
            ################################# LLM Logic ends ##################################################
            ####################################################################################################

            # RRF Search Logic
            movies_data = GetData(Path(__file__).resolve().parents[1]/'data'/'movies.json').get_file_data_json()
            documents = movies_data['movies']

            match args.rerank_method:
                case "individual":
                    # Fetch results 5 times the limit (ranked acc. to RRF)
                    rrf_results = HybridSearch(documents).rrf_search(args.query, args.k, args.limit*5)

                    #########################################################################################
                    if args.debug:                                                        #                 #
                        logger(f"\tRe-Ranking method: {args.rerank_method} ✅SELECTED\n") # Statement:      #
                        logger(f"\tRRF Result:\n{rrf_results}\n")                         #   Logs          #
                    #########################################################################################

                    llm = LlmPrompt('gemma-3-27b-it')
                    results = []
                    for res in rrf_results:

                        res['rerank_score'] = llm.rerank_individual(args.query, res)

                        # Can choose to remove this statement
                        # just a design choice
                        #   but then sort the rrf_results (based on 'rerank_score' key) and print to user
                        results.append(res)

                        time.sleep(1)  #to avoid hitting rate limits (try >=3)

                    results.sort(key=itemgetter('rerank_score'), reverse=True)

                    # Print the result
                    print(f"Re-ranking top {args.limit} results using individual method...")
                    print(f"Reciprocal Rank Fusion Results for '{args.query}' (K={args.k})")
                    for i, result in enumerate(results[:args.limit]):

                        print(f"{i+1}. {result['title']}")
                        print(f"Re-rank Score: {result['rerank_score']:.1f}/10")
                        print(f"RRF Score: {result['rrf_score']}")
                        print(f"BM25 Rank: {result['keyword_rank']}, Semantic Rank: {result['semantic_rank']}")
                        print(f"{result['document']}...\n\n")

                case "batch":
                    # Fetch results 5 times the limit
                    rrf_results = HybridSearch(documents).rrf_search(args.query, args.k, args.limit*5)
                    
                    #########################################################################################
                    if args.debug:                                                        #                 #
                        logger(f"\tRe-Ranking method: {args.rerank_method} ✅SELECTED\n") # Statement:      #
                        logger(f"\tRRF Result:\n{rrf_results}\n")                         #   Logs          #
                    #########################################################################################

                    # Returns a json list
                    response = LlmPrompt('gemma-3-27b-it').rerank_batch(args.query, rrf_results)

                    # Sort the results acc. to Re-Ranking
                    results = []
                    for i, id in enumerate(json.loads(response)):

                        for res in rrf_results:
                            if res['doc_id'] == id:
                                doc = res
                                break
                        doc['rerank_score'] = i+1   # i+1 , as i starts with 0
                        results.append(doc)

                    # Print the result
                    print(f"Re-ranking top {args.limit} results using batch method...")
                    print(f"Reciprocal Rank Fusion Results for '{args.query}' (K={args.k})")
                    for i, result in enumerate(results[:args.limit]):

                        print(f"{i+1}. {result['title']}")
                        print(f"Re-rank Score: {result['rerank_score']:.1f}/10")
                        print(f"RRF Score: {result['rrf_score']}")
                        print(f"BM25 Rank: {result['keyword_rank']}, Semantic Rank: {result['semantic_rank']}")
                        print(f"{result['document']}...\n\n")

                case "cross_encoder":
                    # Fetch results 5 times the limit
                    rrf_results = HybridSearch(documents).rrf_search(args.query, args.k, args.limit*5)
                    
                    #########################################################################################
                    if args.debug:                                                        #                 #
                        logger(f"\tRe-Ranking method: {args.rerank_method} ✅SELECTED\n") # Statement:      #
                        logger(f"\tRRF Result:\n{rrf_results}\n")                         #   Logs          #
                    #########################################################################################

                    # cross encoder re-ranked results
                    results = LlmPrompt('gemma-3-27b-it').rerank_cross_encoder(args.query, rrf_results)
                    results.sort(key=itemgetter('cross_encoder_score'), reverse=True)

                    ##########################################################################################
                    if args.debug:                                                       #                  #      
                        logger(f"Re-Ranking Successful ✅✅✅✅✅✅\n")                # Statement:       #
                        logger(f"Results:\n{results}\n ✅✅✅✅✅✅\n")               #   Logs           #
                    #####################################################################################

                    # Print the result
                    print(f"Re-ranking top {args.limit} results using cross_encoder method...")
                    print(f"Reciprocal Rank Fusion Results for '{args.query}' (K={args.k})")
                    for i, result in enumerate(results[:args.limit]):

                        print(f"{i+1}. {result['title']}")
                        print(f"Cross Encoder Score: {result['cross_encoder_score']:.3f}")
                        print(f"RRF Score: {result['rrf_score']}")
                        print(f"BM25 Rank: {result['keyword_rank']}, Semantic Rank: {result['semantic_rank']}")
                        print(f"{result['document']}...\n\n")

                case _: # Hybrid Search (no Reciprocal Rank Fusion)
                    results = HybridSearch(documents).rrf_search(args.query, args.k, args.limit*5)
 
                    ##########################################################################################
                    if args.debug:                                                       # Statement:       #
                        logger(f"\tResults:\n{results}\n✅✅✅✅✅✅\n")                #   Logs           #
                    ########################################################################################
                    for i, result in enumerate(results[:args.limit]):
                        print(f"{i+1}. {result['title']}")
                        print(f"RRF Score: {result['rrf_score']}")
                        print(f"BM25 Rank: {result['keyword_rank']}, Semantic Rank: {result['semantic_rank']}")
                        print(f"{result['document']}...\n\n")

            if args.evaluate:
                llm_evaluation = LlmPrompt('gemma-3-27b-it').evaluate_result(args.query, results)
        
                ####### For Storing LLM evaluation scores (use this piece of code) ######
                #
                #for idx, llm_score in enumerate(json.loads(llm_evaluation)):
                #    results[idx]['evaluation_score'] = llm_score
                #
                #for i, result in enumerate(results):
                #    print(f"{i+1}. {result['title']} {result['evaluation_score']}/3")
                #########################################################################

                #   Scale:
                #   - 3: Highly relevant
                #   - 2: Relevant
                #   - 1: Marginally relevant
                #   - 0: Not relevant
                llm_scores = json.loads(llm_evaluation)
                ##########################################################################################
                if args.debug:                                                      # Statement:        #
                    logger(f"\tPerforming LLM Evaluation...\n")                    #    Logs           #
                #######################################################################################
                for i, score in enumerate(llm_scores):
                    print(f"{i+1}. {results[i]['title']}: {score}/3")

                    #########################################################################################
                    if args.debug:                                                      # Statement:       #
                        logger(f"\t\t{i+1}. {results[i]['title']}: {score}/3\n")       #   Logs           #
                    ######################################################################################

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()