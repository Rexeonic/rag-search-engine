Inverted Index
--------------
An inverted index is what makes search fast – it's like a SQL database index, but for text search. Instead of scanning every document each time a user searches, we build an index for fast lookup.

    "forward index" maps location -> value. 
    "inverted index" maps value -> location

    eg.

    An inverted index is super fast – we get O(1) lookups on each token. However, building the index is slow, because we have to read every document and tokenize it.

Term Frequency
--------------
Term frequency (TF) measures how often a word appears in a document. 
    
    e.g

    Boots the bear teaches 'students' programming while enjoying his "salmon" for lunch. The 'students' gather around eagerly as they learn debugging techniques, and all 'students' leave with new coding skills.

    students appears 3 times
    salmon appears 1 time

student is likely more important than salmon (which it is, even if the gluttonous bear happens to not agree).