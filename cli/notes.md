(1.0.0) Search
------

    Query: "COVID-19"
    Keyword search: ✅ Finds documents specifically about COVID-19
    Semantic-only search: ❌ Finds less-related content about "respiratory viruses" or "pandemic diseases"
    

    Query: "Alien movie"
    Keyword search: ❗Only finds "Alien"
    Semantic-only search: ✅ Finds similar movies like "Invaders from Mars", "E.T.", or "Alien"

(1.0.1)Text Processing Pipeline
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

