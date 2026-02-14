Inverted Index
--------------
An inverted index is what makes search fast – it's like a SQL database index, but for text search. Instead of scanning every document each time a user searches, we build an index for fast lookup.

    "forward index" maps location -> value. 
    "inverted index" maps value -> location

    eg.

    An inverted index is super fast – we get O(1) lookups on each token. However, building the index is slow, because we have to read every document and tokenize it.