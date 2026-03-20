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


Vector Addition
---------------

Vector addition is useful for combining/mixing concepts.
example, "I want a result that's like this and like that."

With vector addition, we just add the corresponding elements of two vectors together.

    [0.5, -0.2,  0.8]
    +
    [0.1,  0.9, -0.3]
    =
    [0.6,  0.7,  0.5]

"The Great Bear" embeds to [0.5, -0.2, 0.8] (Animation, Adventure, Family)
                            +
"Back Country" which embeds to [0.1, 0.9, -0.3] (Horror, Thriller, Survival)
                            =
[0.6, 0.7, 0.5], which represents a mix of those genres (Family, Adventure, Horror).

Vector Subtraction
------------------

Vector subtraction is useful for removing concepts.
example, "I want a result that's like this but not that."

    [0.5, -0.2,  0.8]
    -
    [0.1,  0.9, -0.3]
    =
    [0.4, -1.1,  1.1]

"The Great Bear" embeds to [0.5, -0.2, 0.8] (Animation, Adventure, Family)
                                -
"The Revenant," which embeds to [0.1, 0.9, -0.3] (Horror, Thriller, Survival)
                                =
[0.4, -1.1, 1.1], representing Family Adventure but not Horror or Survival.


Locality-Sensitive Hashing
--------------------------
 pre-group similar vectors into "buckets" using a special hash function. 

    It speeds up searches but can miss some
    similar vectors ( i.e `lower recall`)

will use if computation speed is a priority
over perfect accuracy.

Vector Database
----------------
is designed specifically for storing and searching high-dimensional vectors efficiently. They offer:

•Fast similarity search: Sub-linear time complexity using indexing
•Persistent storage: Embeddings saved to disk
•Distributed architecture: Handle more data than a single machine can store in RAM
•Concurrent access: Multiple users can search simultaneously

   +---------------------------------------------------------------------------------------------------+ 
   |    Traditional Database 	            |                  Vector Database                         |
   +---------------------------------------------------------------------------------------------------+ 
   | Data: Structured (rows/columns) 	    |     Data: High-dimensional vectors                       | 
   | Queries: Exact matches (WHERE clauses) |	Queries: Similarity search (nearest neighbors)         | 
   | Use Case: Transactional data 	        |    Use Case: ML embeddings, semantic search              | 
   +---------------------------------------------------------------------------------------------------+ 

Vector databases also use specialized indexing techniques to speed up similarity searches, such as:

•HNSW: Hierarchical navigable small world
•IVF: Inverted File Flat Vector
•LSH: Locality-sensitive hashing


PGVector -> open source (for PostgreSQL)
sqlite-vec -> open source (for SQLite)

