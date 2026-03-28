# Standard Library
import os

# External Dependencies
from sentence_transformers import CrossEncoder
from google import genai
from google.genai import types
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

        return self._response(prompt).text
    
    def rewrite(self, query):
        """
        Pre-search query function for query rewriting
        """
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

        return self._response(prompt).text
    
    def expand(self, query):
        """
        Pre-search query function for query expansion
        """
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
        
        return self._response(prompt).text
    
    ############## FUNCTION for RERANKING ##############
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
        
        score = self._response(prompt).text

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
        
        ranking = self._response(prompt).text

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
    ############## Function for RERANKING ends ###############

    ############## Evalution of Search Results ###############
    def evaluate_result(self, query, results):
        """
            Use LLM to Evaluate the Search Result generated by
            data retrieval.

            chr(10) == \n
        """
        formatted_results = []
        for result in results:
            formatted_results.append(f"{result['title']}: {result['document']}")

        prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:
                {chr(10).join(formatted_results)}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers other than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]"""
        
        llm_evaluation = self._response(prompt).text

        return llm_evaluation

    ########### Functions for AUGMENTED GENERATION ##############
    def llm_augmentation(self, query, search_results):

        template = '''<movie id={id}>{title}:\n{desc}\n</movie>\n'''
        docs = ""
        for movie in search_results:
            docs += template.format(id=movie['doc_id'], title=movie['title'], desc=movie['document'])

        prompt = f"""You are a RAG agent for a movie streaming service.
            Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
            Provide a comprehensive answer that addresses the user's query.

            Query: {query}

            Documents:
            {docs}

            Answer:"""
        
        rag_response = self._response(prompt).text

        return rag_response
    
    def llm_summarization(self, query, results):
        prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

            This should be tailored to movie enthusiasts who are OTT users. OTTs are streaming platforms such as Netflix, Amazon Prime,
            Warner Bros, Disney etc..

            Query: {query}

            Search results:
            {results}

            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
        
        return self._response(prompt).text

    def llm_citation(self, query, search_result):

        template = '''<movie id={id}>{title}:\n{desc}\n</movie>\n'''
        documents = ""
        for movie in search_result:
            documents += template.format(id=movie['doc_id'], title=movie['title'], desc=movie['document'])

        prompt = f"""Answer the query below and give information based on the provided documents.

            The answer should be tailored to users of a movie streaming service.
            If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

            Query: {query}

            Documents:
            {documents}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources in the format [1], [2], etc. when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the provided documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""
        
        return self._response(prompt).text

    def llm_qna(self, question, search_result):

        template = '''<movie id={id}>{title}:\n{desc}\n</movie>\n'''
        context = ""
        for movie in search_result:
            context += template.format(id=movie['doc_id'], title=movie['title'], desc=movie['document'])

        prompt = f"""Answer the following question based on the provided documents. If documents are not enough to answer
            sear

            Question: {question}

            Documents:
            {context}

            General instructions:
            - Answer directly and concisely
            - Use only information from the documents
            - If the answer isn't in the documents, search web and relevant sources
            - If still could not find answer, say 'I don't have enough information to provide an answer'
            - Cite sources when possible

            Guidance on types of questions:
            - Factual questions: Provide a direct answer
            - Analytical questions: Compare and contrast information from the documents
            - Opinion-based questions: Acknowledge subjectivity and provide a balanced view

            Answer:"""
        
        return self._response(prompt).text
    
    def image_search(self, query, image_content, mime):
        """
        :params image_content: image file in bytes
        :params mime:   media type (similar to file format)
        """
        system_prompt = f"""Given the included image and text query, rewrite the text query to improve search results from a movie database.

            Make sure to:
                - Synthesize visual and textual information
                - Focus on movie-specific details (actors, scenes, style, etc.)
                - Return only the rewritten query, without any additional commentary"""
        
        parts = [
            system_prompt,
            types.Part.from_bytes(data=image_content, mime_type=mime),
            query.strip()
        ]

        return self._response(parts)
    
    def _response(self, prompt):
        response = self.client.models.generate_content(
            model = self.model, 
            contents = prompt
        )

        return response