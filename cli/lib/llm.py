# Standard Library
import os

# External Dependencies
from sentence_transformers import CrossEncoder
from google import genai
from dotenv import load_dotenv

class LlmPrompt:
    def __init__(self, model):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set")


        self.client = genai.Client(api_key=api_key)
        self.model = model

    def spell(self, query):
        """
        Pre-search query function for spelling correction
        """
        prompt = f"""Fix any spelling errors in the user-provided movie search query below.
                    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
                    Preserve punctuation and capitalization unless a change is required for a typo fix.
                    If there are no spelling errors, or if you're unsure, output the original query unchanged.
                    Output only the final query text, nothing else.
                    User query: "{query}"
                  """

        return self._response(prompt)
    
    def rewrite(self, query):
        prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.
                            Consider:
                            - Common movie knowledge (famous actors, popular films)
                            - Genre conventions (horror = scary, animation = cartoon)
                            - Keep the rewritten query concise (under 10 words)
                            - It should be a Google-style search query, specific enough to yield relevant results
                            - Don't use boolean logic

                            Examples:
                            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                            If you cannot improve the query, output the original unchanged.
                            Output only the rewritten query text, nothing else.

                            User query: "{query}"
                """

        return self._response(prompt)
    
    def expand(self, query):
        prompt = f"""Expand the user-provided movie search query below with related terms.

                        Add synonyms and related concepts that might appear in movie descriptions.
                        Keep expansions relevant and focused.
                        Output only the additional terms; they will be appended to the original query.

                        Examples:
                        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                        - "action movie with bear" -> "action thriller bear chase fight adventure"
                        - "comedy with bear" -> "comedy funny bear humor lighthearted"

                        User query: "{query}"
                """
        
        return self._response(prompt)
    
    def rerank_individual(self, query, result):
        prompt = f"""Rate how well this movie matches the search query.

                                Query: "{query}"
                                Movie: {result.get('title', '')} - {result.get('document', '')}

                                Consider:
                                - Direct relevance to query
                                - User intent (what they're looking for)
                                - Content appropriateness

                                Rate 0-10 (10 = perfect match).
                                Output ONLY the number in your response, no other text or explanation.

                                Score:"""
        
        score = self._response(prompt)

        return int(score) if score != None else 0

    def rerank_batch(self, query, movie_list):

        template = '''<movie id={id}>{title}:\n{desc}\n</movie>\n'''
        movie_list_str = ""
        for movie in movie_list:
            movie_list_str += template.format(id=movie['doc_id'], title=movie['title'], desc=movie['document'])

        prompt = f"""Rank the movies listed below by relevance to the following search query.

                    Query: "{query}"

                    Movies:
                    {movie_list_str}

                    Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

                    For example:
                    [75, 12, 34, 2, 1]

                    Ranking:"""
        
        ranking = self._response(prompt)

        return ranking
 
    def rerank_cross_encoder(self, query, movie_list):
        pairs = []
        for movie in movie_list:
            pairs.append([query, f"{movie.get('title', '')} - {movie.get('document', '')}"])

        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

        # `predict` returns a list of numbers, one for each pair
        scores = cross_encoder.predict(pairs)

        for idx, movie in enumerate(movie_list):
            movie['cross_encoder_score'] = scores[idx]

        return movie_list
    def _response(self, prompt):
        response = self.client.models.generate_content(
            model=self.model, 
            contents=prompt
        )

        return response.text