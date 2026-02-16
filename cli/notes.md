(1.0) Search
------

    Query: "COVID-19"
    Keyword search: ✅ Finds documents specifically about COVID-19
    Semantic-only search: ❌ Finds less-related content about "respiratory viruses" or "pandemic diseases"
    

    Query: "Alien movie"
    Keyword search: ❗Only finds "Alien"
    Semantic-only search: ✅ Finds similar movies like "Invaders from Mars", "E.T.", or "Alien"

(1.1)Text Processing Pipeline
------------------------
![alt text](text_pre_processing.png)

Case insensitivity: Convert all text to lowercase

    "The Matrix" becomes "the matrix"
    "HE IS HERE" becomes "he is here"

Remove punctuation: We don't care about periods, commas, etc

    "Hello, world!" becomes "hello world"
    "sci-fi" becomes "scifi"

Tokenization: Break text into individual words

    "the matrix" becomes ["the", "matrix"]
    "hello world" becomes ["hello", "world"]

Stop words: Remove common stop words that don't add much meaning

    ["the", "matrix"] becomes ["matrix"]
    ["a", "puppy"] becomes ["puppy"]

Stemming: Keep only the stem of words
    ![alt text](stemming.png)
    
    ["running", "jumping"] becomes ["run", "jump"]
    ["watching", "windmills"] becomes ["watch", "windmill"]

(2.0)Inverted Index
-------------------
An inverted index is what makes search fast – it's like a SQL database index, but for text search. Instead of scanning every document each time a user searches, we build an index for fast lookup.

    "forward index" maps location -> value. 
    "inverted index" maps value -> location

    eg.

    An inverted index is super fast – we get O(1) lookups on each token. However, building the index is slow, because we have to read every document and tokenize it.

(2.1)Term Frequency
-------------------
Term frequency (TF) measures how often a word appears in a document. 
    
    e.g

    Boots the bear teaches 'students' programming while enjoying his "salmon" for lunch. The 'students' gather around eagerly as they learn debugging techniques, and all 'students' leave with new coding skills.

    students appears 3 times
    salmon appears 1 time

student is likely more important than salmon (which it is, even if the gluttonous bear happens to not agree).

(2.2)Inverted Document Frequency (IDF)
--------------------------------------
    
Document Frequency (DF) ->  measures how many documents in the dataset contain a term. 
                            The more documents a term appears in, the bigger its value.

Why Inverse?
ans) because we want rare terms to have higher scores. Say we have 100 documents in total:

eg. let's say there are 100 documents

Common term (appears in 95 documents):

    bear: IDF = log(100/95) = 0.02 ← Low score

Rare term (appears in 2 docs):

    cyborg: IDF = log(100/2) = 1.7 ← High score

Universal term (appears in all docs):

    movie: IDF = log(100/100) = 0 ← Zero score



    1. TF (Term Frequency): How often a term appears in a document
    2. IDF (Inverse Document Frequency): How rare a term is across all documents

(2.3)TF-IDF (Term Frequency-Inverse Document Frequency)
-------------------------------------------------------
   +-------------------+
   | TF-IDF = TF * IDF |
   +-------------------+


    Frequent words get high TF scores
    --------
    Rare words get high IDF scores
    ----------
    Best matches have both high TF and high IDF
    ------------
