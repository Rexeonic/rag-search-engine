"""
Docstring for cli.parameters

Dedicated module to place constants , so they can
be reused across different parts of codebase
"""

#       BM25 Parameters
# tunable parameter that controls the diminishing returns
# diminishing returns - after a certain point, additional occurrences matter less
BM25_K1 = 1.5    # a common value is 1.5
BM25_B = 0.75

# controls the precision of semantic search
#   rounds of the semantic score upto the
#   given value
SCORE_PRECISION = 4


# controls the ratio
#       results from keyword search & results
#       from semantic search, should be in what
#       proportion in the final result.
# 
#   alpha = 1.0 (100% keyword search)
#   alpha = 0.0 (100% semantic search) 
ALPHA = 0.5


# RRF K parameter:
# ---------------
#       rrf_score = 1 / (k + rank)
# controls weight given to higher-ranked results vs. Lower Ranked results
#
# Lower 'k' value (eg. 20): more weight to top ranked results
# Higher 'k' value (eg. 100): more influence to lower ranked results
RRF_K = 60