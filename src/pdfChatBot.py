import requests
import subprocess
from tqdm import tqdm
from elasticsearch import Elasticsearch
import json
import nltk
from nltk.tokenize import sent_tokenize


# Connect to ElasticSearch running on localhost
es = Elasticsearch(["http://localhost:9200"])

class pdfChatBot():

    def __init__(self, index_name, model_name="gemma:2b"):
        self.model_name = model_name
        self.index_name = index_name

        self.es_url = "http://localhost:9200"

        # create es index
        self.create_es_index()
        # run Ollama model
        self.run_ollama_model()

    def run_ollama_model(self):
        """
        Run the Ollama model inside of the container.
        """
        # Run the docker command and capture output
        result = subprocess.run(
            ["docker", "exec", "-i", "ollama", "ollama", "run", self.model_name], 
            capture_output=True, 
            text=True
        )

        # Check the output or error
        print("Output:", result.stdout)
        print("Error:", result.stderr)

    def create_es_index(self):
        """
        Create an index for ElasticSearch.
        """
        # Define the ElasticSearch URL
        es_url = "http://localhost:9200"

        # Define the index mapping for ElasticSearch
        mapping = {
            "mappings": {
                "properties": {
                    "text": { "type": "text" },
                    "embedding": { "type": "dense_vector", "dims": 2048 }  # Adjust dims as per your model's embedding size
                }
            }
        }

        # Create the index
        response = requests.put(f"{es_url}/{self.index_name}", json=mapping)

        # Check the response
        if response.status_code == 200:
            print(f"Index '{self.index_name}' created successfully.")
        else:
            print(f"Failed to create index: {response.json()}")

    def fetch_document_from_es(self):
        """
        Fetch a document from the ElasticSearch
        """
        es_url = f"http://localhost:9200/{self.index_name}/_search"

        search_query = {
            "size": 5,
            "query": {
                "match_all": {}
            }
        }

        response = requests.get(es_url, json=search_query)
        return response.json()

    def check_index_exists(self):
        """Checks if an Elasticsearch index exists."""

        url = f"http://localhost:9200/{self.index_name}"
        response = requests.head(url)

        if response.status_code == 200:
            return True
        elif response.status_code == 404:
            return False
        else:
            raise Exception(f"Unexpected status code: {response.status_code}")

    def semantically_break_text(self, pdf_text):
        """
        Use nltk to break up text into sentences and combine 
        sentences into paragraphs of about 100 tokens.
        """
        # nltk.download("punkt")  # Download sentence tokenizer if needed
        # sentences = sent_tokenize(pdf_text)

        import re
        sentences = re.split(r'(?<=[.!?]) +', pdf_text)

        chunk_size = 100  # Approximate word count per chunk
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += sentence_length

        # Append any remaining sentences as the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def index_pdf_paragraphs(self, pdf_text):
        """
        Break PDF document up into paragraphs or sections of
        around 100-200 words and create vector embeddings.
        """
        # split by paragraph
        chunks = self.semantically_break_text(pdf_text)
        # iterate paragraphs
        for chunk in tqdm(chunks):
            if chunk:  # Skip empty lines
                # generate embeddings from ollama
                embedding = self.generate_embedding(chunk)
                # store embedding in es
                self.store_paragraph_in_es(chunk, embedding)

    def store_paragraph_in_es(self, paragraph_text, embedding):
        # ElasticSearch URL
        es_url = f"http://localhost:9200/{self.index_name}/_doc"

        data = {
            "text": paragraph_text,
            "embedding": embedding
        }
        response = requests.post(es_url, json=data)
        return response.json()

    def generate_embedding(self, text):
        # Define the URL and payload
        url = "http://localhost:11434/api/embed"
        payload = {
            "model": self.model_name,
            "input": text
        }

        # Make the POST request
        response = requests.post(url, json=payload)

        # Check the response
        if response.status_code == 200:
            # Print the JSON response if the request was successful
            return response.json()['embeddings'][0]
        else:
            print(f"Request failed with status code {response.status_code}")
            return response.text

    def search_similar_paragraphs(self, query_text, top_n=5):
        """
        Return top 5 vector embeddings from ElasticSearch index.
        """
        es_url = f"http://localhost:9200/{self.index_name}/_search"

        # Generate the embedding for the query text
        query_embedding = self.generate_embedding(query_text)

        # Define the ElasticSearch search query
        search_query = {
            "size": top_n,  # Top n results
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {
                            "query_vector": query_embedding
                        }
                    }
                }
            }
        }

        # Send the search request to ElasticSearch
        response = requests.post(es_url, json=search_query)
        response.raise_for_status()  # Raise an error if the request failed

        # Return the top 5 search results
        return response.json()["hits"]["hits"]

    def reset_es_index(self):
        """
        Delete and remake ElasticSearch index.
        """
        # Define the ElasticSearch URL and index name
        es_url = "http://localhost:9200"  # Localhost for your Docker container

        # Make the DELETE request
        response = requests.delete(f"{es_url}/{self.index_name}")

        # Check if the deletion was successful
        if response.status_code == 200:
            print(f"Index '{self.index_name}' deleted successfully.")
        else:
            print(f"Failed to delete index '{self.index_name}':", response.json())

        # re-create es index
        self.create_es_index()

    # Function to format ElasticSearch results
    def format_context(self, hits):
        context = "Top 5 relevant documents:\n"
        for i, hit in enumerate(hits, 1):
            context += f"\nDocument {i}:\n{hit['_source']['text']}\n"  # Adjust 'text_field' as needed
        return context

    def generate_response(self, query):
        """
        Generate reponse using RAG.
        """
        # Ollama chat url
        ollama_url = "http://localhost:11434/api/chat"
        # get top hits from is
        top_hits =  self.search_similar_paragraphs(query)
        # Format the context for Ollama
        context = self.format_context(top_hits)
        # format prompt with context and query
        prompt = f"{context}\n\nQuery: {query}"
        # format payload
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        # Send the request to Ollama
        response = requests.post(ollama_url, json=payload)
        # response.raise_for_status()

        # Initialize an empty string to gather the full response content
        full_response = ""

        # Process each line of the streaming response
        for line in response.iter_lines():
            if line:  # Only process non-empty lines
                try:
                    # Parse the line as JSON
                    line_data = json.loads(line.decode("utf-8"))
                    # Concatenate the content field if it's present
                    if "message" in line_data and "content" in line_data["message"]:
                        full_response += line_data["message"]["content"]
                except json.JSONDecodeError as e:
                    print("Error decoding line:", e)

        # Print the fully assembled response
        return full_response
        # return response.json()# ["response"]

    def generate_response_without_rag(self, query):
        """
        Generate response without RAG.
        """
        # Ollama chat url
        ollama_url = "http://localhost:11434/api/chat"
        # format prompt with context and query
        prompt = f"Query: {query}"
        # format payload
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        # Send the request to Ollama
        response = requests.post(ollama_url, json=payload)
        # response.raise_for_status()

        # Initialize an empty string to gather the full response content
        full_response = ""

        # Process each line of the streaming response
        for line in response.iter_lines():
            if line:  # Only process non-empty lines
                try:
                    # Parse the line as JSON
                    line_data = json.loads(line.decode("utf-8"))
                    # Concatenate the content field if it's present
                    if "message" in line_data and "content" in line_data["message"]:
                        full_response += line_data["message"]["content"]
                except json.JSONDecodeError as e:
                    print("Error decoding line:", e)

        # Print the fully assembled response
        return full_response


