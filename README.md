Overview
--------
    rag-search-engine is implemented as a collection of cmd line
    scripts that performs various search operations on a local
    dataset of movies (currently)

Index:
        <ul>
            <li>Architecture</li>
            <ol>
                <li> Entrypoint </li>
                <li> Keyword Search </li>
                <li> Okapi BM25 </li>
                <li> Semantic Search </li>
                <li> Chunked Semantic Search </li>
            </ol>
        </ul>

<h2>Architecture</h2>

Entrypoint
----------
cli.keyword_search_cli.py is one entrypoint

    Dependencies :
        1) cli.lib.preprocessing (local)
        2) cli.lib.inverted_index (local)
        3) argparse (library)
    
cli.semantic_search_cli.py is other entrypoint (for semantic search)

    Dependencies :
        1) cli.lib.semantic_search  (local)
                |
                +--> SentenceTransformer (imports `all-MiniLM-L6-v2` model)
                +--> Numpy
                +--> from lib.preprocessing GetData class
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

Practically, deduce<br>
    If a term is rare across all documents ( i.e IDF value is high),
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
            0.5 -> Additive/Laplace Smoothing
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

<h5>Cosine Similarity</h5>   (all-MiniLM-L6-v2 uses cosine similarity)
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

Preprocessing for Embeddings
----------------------------
It is already:
    1. Case insensitive
    2. Punctuation robust
    3. Whitespace tolerant

just basic cleaning like stripping whitespace is okay

Keep same model for both data (i.e documents) and 
queries. (as diff. model learns different mathematical space)

Chunking
--------

overlap -> create chunks that share words to 
           preserve context across boundries.
           
(about 20% overlap works for most use cases, `but test on dataset`)
eg: 
In the climatic scene, the bear attack was
bear attack was terrifying. The stunning and 
The stunning and innovative special effects.


Semantic Chunking
-----------------
<b>Semantic chunking</b> respects natural language structure
like sentences and paragraphs. Each chunk contains <i>complete
thought</i> i.e split at natural breaks like sentences or paragraphs.


`can still use overlap with semantic chunking.`
Semantic chunking with overlap works well for most situations
(but there are much advanced techniques like 
"Colbert", "Late Chunking")

<dl>
<dt><h4><u>Colbert (pronounced "cole-bear")<u></h4></dt>
<dd>creates one embedding per word, with each word contextualized:

    "Ted" (the main character)
    "comedy" (the genre)
    "John" (the human character)
    "responsibilities" (related to growing up)

It is an example of multi-vector retrieval (MVR), where a document or chunk is represented by multiple vectors (e.g., one per token) rather than a single vector per chunk

    trade-off: ColBERT requires more storage and computational power
</dd>
<dt><h4><u>Late Chunking<u></h4></dt>
<dd>creates an embedding for the entire document (or as much of it as possible), and then uses that embedding to create context-aware embeddings for each word

So, each word contributes more meaningful information to the final embedding because its role in the text is already understood.
</dd>
</dl>

<h5>When to use advanced Techniques?<h5>
<ul type='disc'>
    <li>need extremely precise search results.</li>
    <li>Standard approaches aren't meeting accuracy requirements.</li>
    <li>working with complex, nuanced text where context is critical.</li>

Chunked Semantic Search
-----------------------
1. Searching across chunks using cosine similarity with query embeddings
2. Aggregating chunk scores to determine the most relevant documents
3. Returning formatted results that map chunks back to their original movies

    Note :-
        By searching at the chunk level, we can find relevant information even when it's buried deep within a long document, like a book or technical manual.

Hybrid Search
-------------

1. Score Normalization
    BM25: 0–100+    (keyword search)
    Cosine: 0–1     (semantic search)

    <u><b>Min-Max Normalization</b> (<i>Used</i>)</u>
    Normalized score = (score - min_score) / (max_score - min_score)

2. Weighted Combination
    alpha ("α") -> dynamically controls the weighting between the two scores<br>

    α = 1.0: [████████████████████] 100% Keyword<br>
    α = 0.7: [██████████████------] 70% Keyword, 30% Semantic<br>
    α = 0.5: [██████████----------] 50/50 Split<br>
    α = 0.2: [████----------------] 20% Keyword, 80% Semantic<br>
    α = 0.0: [--------------------] 100% Semantic<br>

    
    Hybrid Score =  (alpha * bm25_score) + (1 - alpha) * semantic_score

3. Reciprocal Rank Fusion

    rrf Score = 1 / (k + rank)

        where,
            k -> weights given to higher ranked vs. lower ranked results
            on avg. 40 < k > 60 (but can be configured accordingly)

        Lower 'k' value (eg. 20): more weight to top ranked results
        Higher 'k' value (eg. 100): more influence to lower ranked results
        
    uses ranking instead of score (no need for normalization).

    Working:

    • BM25<br><br>
        Brother Bear (15.2) = 1 / (60 + 1) = 0.0164<br>
        Jungle Book (6.3)   = 1 / (60 + 2) = 0.0161<br>
        Paddington (8.7)    = 1 / (60 + 3) = 0.0159<br>

    • Semantic<br><br>
        Paddington (0.8)    = 1 / (60 + 1) = 0.0164<br>
        Brother Bear (0.7)  = 1 / (60 + 2) = 0.0161<br>
        We Bear Bears (0.6) = 1 / (60 + 3) = 0.0159<br>

    So,

     Brother Bear = 0.0164 + 0.164 = 0.0325<br>
     Paddington = 0.0161 + 0.0161 = 0.0323<br>
     Jungle Book = 0.0161<br>
     We Bare Bears = 0.0159<br>

Re-Ranking
----------
Search will find 100 movies, but users only look at top 5.

Solution: Two-Stage Search
1. Stage 1 -> Fast BM25/cosine similarity search finds the 
              top ~25 documents

2. Stage 2 -> Slow re-ranking finds the best 5 from 
              those ~25 candidates


It's accurate, but it's <b>much slower</b> - can't pre-cache 
anything.

    Used 2 types of Re-Ranking
    1. Individual

        calling the LLM individually for each document and scoring them.

    2. Batch

        calling the LLM individually for each one can be slow & expensive.

        Quality of search suffers when we score each document independently
        on an arbitrary scale. <b>By giving all the documents as a batch, 
        they're always be compared against each other in the same context.</b>

Semantic search embeddings were created with a bi-encoder (embeds document & query separately so we can use cosine similarity to score)

Another Re-Ranking (based on cross-encoding i.e taking query & document embedding as a single pair)
    3. Cross Encoding
        used `from SentenceTransformers import CrossEncoder`

        cross_encoder -> ms-marco-TinyBERT-L2-v2

        catch subtle errors that bi-encoders miss

        Cross Encoder are usually <u>a Regression model</u> which can be fine-tuned
        to one's need.

Manual Evaluation
-----------------
The kind of ways the search can fail needs to be predicted and tackled accordingly.
So, manually testing search is more important than it looks.

In Manual Evaluation, will create `Golden Dataset` with the help of some industry 
expert. 

Think like:
    What's here that shouldn't be?<br>
    What's not here that should be?<br>
    Would I click on these results?<br>

    Many technically correct results are not ideal (as subconcious thinks these are wrong.)
    These needs to be tackled as well.

1. Did the result give you enough information to know whether the movies are relevant?<br>
2. Would it be better to return fewer results because the last few usually aren't relevant?<br>
3. Would it be better to return more results because there are more highly relevant options that just missed the cutoff?<br>


For Manual Evaluation
    run cli/evaluation_cli.py

    e.g
    uv run cli/evaluation_cli.py --limit Z

        where,
             Z belongs to set of integers

LLM Evaluation
--------------

To let an LLM judge, you must:

  •  Define clear evaluation criteria
  •  Specify what makes a result relevant
  •  Articulate your quality standards

This clarity improves your entire evaluation system, even if you end up not using the LLM.

Implementation Strategy

1. Start with experts – Define clear evaluation criteria<br>
2. Create detailed prompts – Include domain knowledge<br>
3. Validate on samples – Check that the LLM agrees with experts<br>
4. Use for scale – Let the LLM handle bulk evaluation<br>
5. Spot-check results – Have experts review surprising scores<br>


For LLM Evaluation
    run hybrid_search_cli.py rrf_search with --evaluate option

    eg. 
        uv run cli/hybrid_search_cli.py rrf_search "query" --evaluation