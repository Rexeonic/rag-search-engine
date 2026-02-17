"""
Docstring for cli.parameters

Dedicated module to place constants , so they can
be reused across different parts of codebase
"""

# tunable parameter that controls the diminishing returns
# diminishing returns - after a certain point, additional occurrences matter less
BM25_K1 = 1.5    # a common value is 1.5