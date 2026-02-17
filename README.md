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
    

Working
-------

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
BM25 uses a more stable IDF formula:

    IDF = log((N - df + 0.5) / (df + 0.5) + 1)

    where,
        N = total number of documents in the collection
        df = document frequency (how many documents contain this term)
        0.5 -> Additive/Laplace Smooting
        +1 -> so IDF is always positive (handles some edge cases)

Advantages:
a) Better IDF calculation
b) Term frequency saturation
c) Document length normalization