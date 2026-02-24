Overview
    rag-search-engine is implemented as a collection of cmd line
    scripts that performs various search operations on a local
    dataset of movies (currently)

Entrypoint
----------
rag-search-engine.cli.keyword-search-cli is the main entrypoint

    Dependencies :
        1) cli.lib.preprocessing (local)
        2) cli.lib.inverted_index (local)
        3) argparse (library)
    

Keyword Search
--------------

    1. Text Preprocessing
       
        cli.lib.preprocessing module contains Preprocessing class.

        which is a pipeline through which text comes out 
        Stemmed ( in its real & base form )

    
    2. TF-IDF ( Term Frequency-Inverse Document Frequency )

        caches are built (serialized pickle files) 
        @ rag-search-engine.cache 

        index.pkl  -> This contains index (document id's) of movie
                      in which certain token is present.

            eg.  424 : We're Bear Bears     (document object)
                 1136: Jumanji              (document object)
                 326 : Jungle Bears         (document object)

                 then, index.pkl contains

                 bear : [326,424]   ( token --> doc_id/index )

        docmap.pkl  -> maps index (document id's) with actual 
                       document object.

            eg.  0 ---> None
                 .
                 .
                326 --> Jungle Bears (movie object)
                 .
                424 --> We're Bear Bears
                 .
                1136 -> Jumanji      (movie object)

        term_frequencies.pkl -> contains data of token frequency
                                in a given document object

            eg. 424 : We're Bear Bears (document object)

                424 --> bear = 2 we = 1 ...

Why did all this ?

    Term Frequency score: tells how rare or common a token is in the
    document object

    Inverse Document Frequency score: tells how rare or common a token is
    across all the document objects.

Practically, deduce
    a) If a term is rare across all documents ( i.e IDF value is high),
       and, it is common in a particular document (i.e TF value is high)

       we've got the "Match".

    can determine for rest of the cases as well....

Mathematically, it can be implemented as product of TF & IDF

    let user searches - "Future Cyborg"

    Document 1: A Traveller from future created by future John connor

        cyborg: TF=0, IDF=2.9 (i.e rare)  = 0*2.9 =0
        future: TF=2, IDF=0.05(i.e common)= 0.1
        Total = 0.1

    Document 2: John Connor & cyborg friend
        cyborg:   TF=1, IDF=2.9  = 2.9
        future:   TF=0, IDF=0.05 = 0
        Total = 2.9
    
    Document 3: The Terminator - A cyborg from future
        cyborg: TF=1, IDF=2.9 = 2.9
        future: TF=1, IDF=0.05 = 0.05
        Total = 2.95

Result of Search  (retrieves)  
    1. The Terminator - A cyborg from future 
    2. John Connon & cyborg friend


This technique in search is "TF-IDF" (pre-google era technique)
It is not as robust as today's need that's why we implement


Okapi BM25
----------
 a) BM25 uses a more stable IDF formula:
    -----------------------------------

        IDF = log((N - df + 0.5) / (df + 0.5) + 1)

        where,
            N = total number of documents in the collection
            df = document frequency (how many documents contain this term)
            0.5 -> Additive/Laplace Smooting
            +1 -> so IDF is always positive (handles some edge cases)

 b) Term Frequency Problem
    ----------------------
    If a word appears 100 times, it gets 10x more weight than a word that appears 10 times.
    This creates problems

    Query: "bear hunting"

        Document A: "bear bear bear bear" → 4 matches
        Document B: "bear hunting guide for beginners" → 2 matches

    With basic TF, Document A gets a much higher score despite being clearly less useful!

        Solution
        --------
            BM25 uses diminishing returns – after a certain point,
            additional occurrences matter less.

            tf_component = (tf * (k1 + 1)) / (tf + k1)

        +---------------+----------+-----------------+
        |Term Frequency |Basic TF  | BM25 TF (k1=1.5)|
        +---------------+----------+-----------------+
        |     1         |     1    |       1.0       |
        |     2         |     2    |       1.4       |
        |     5         |     5    |       1.9       |
        |     10        |     10   |       2.2       |
        |     20        |     20   |       2.3       |
        +---------------+----------+-----------------+

 c) Document length Normalization
    -----------------------------
    ensuring longer documents don't get unfair advantages over shorter, more focused ones. 
    Longer documents contain more words, which can artificially boost their scores

    # Length normalization factor
    length_norm = 1 - b + b * (doc_length / avg_doc_length)

    # Apply to term frequency
    tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
Advantages:
a) Better IDF calculation
b) Term frequency saturation
c) Document length normalization


Semantic Search
---------------

    cli.lib.semantic_search.py [where logic is implemented]

fundamental tool that will power our semantic search is "embeddings" 
(numerical representations of text that capture the meaning of words)

Semantic embeddings are usually in over 300 dimensions.

    "embedding", when we take a piece of text and convert it into a vector. 
    eg.

        "King" -> [3.5, 2.5]
        "Queen" -> [3.0, 2.0]
        "Human" -> [3.0, -3.0]

    distance b/w the vectors represents how similar the meanings of the words are. 

The process of converting text into vectors requires a lot of data and computation
it's a machine learning "training" process

    --------------------------------------------------------------
    | Used a pre-trained embedding model called all-MiniLM-L6-v2 |
    --------------------------------------------------------------


    General Purpose Models

        Use case: Broad semantic understanding across domains
        Examples: all-MiniLM-L6-v2, all-mpnet-base-v2
        Best for: Movie search, general document retrieval

    Domain-Specific Models

        Use case: Specialized knowledge (medical, legal, scientific)
        Examples: allenai-specter, microsoft/BiomedNLP-PubMedBERT
        Best for: Technical documentation, research papers

    Multilingual Models

        Use case: Data in multiple languages in the same search system
        Examples: paraphrase-multilingual-MiniLM-L12-v2
        Best for: International movie databases

Dot Product
-----------

dot product measures how much two vectors "point in the same direction.

a) more similar the vectors -> higher dot product
b) point in opposite directions -> dot product will be negative.

    problem
    1. affected by vector magnitude, whereas
    direction is the important part for semantic similarity

    Note:
    Vectors has, 
    1. magnitude -> represents 'confidence' or 'strength'
    2. direction -> semantic similarity

Cosine Similarity   (all-MiniLM-L6-v2 uses cosine similarity)
-----------------
    ![alt text](resources/cosine_similarity.png)

     measures the cosine of the angle between two vectors, 
     meaning it only cares about their direction.

range -> -1.0 to 1.0

    1.0 - vectors point in exactly the same
          direction (perfectly similar)

    0.0 - vectors are perpendicular
          (not similar)

    -1.0 - vectors point in opp. directions
           (perfectly dissimilar)
     

Formula
   ------------------------------------------------------------------------- 
   | cosine_similarity = dot_product(A, B) / (magnitude(A) × magnitude(B)) |
   ------------------------------------------------------------------------- 

Mechanics,
    Calculate similarity: The dot product measures how much vectors align
    Remove length bias: Dividing by magnitudes removes the effect of vector size


****Note****
    Use same similarity as to which the embedding model was trained on.


    all-MiniLM-L6-v2 was trained on cosine similarity i.e it is used

    