# Retrieval Augmented Generation with Ollama and ElasticSearch

Results comparing chat bot responses with and without RAG can be found in notebooks/chat-with-a-pdf.ipynb.

All source code can be found in src/pdfChatBot.py.

Application can be deployed by running `docker compose up`.

For pulling and running the Ollama image instructions are below:

### 1. Pull image
docker pull ollama/ollama

### 2. Run container in detached mode
docker run -d ollama/ollama

### 3. Exec into container
docker exec -it ollama ollama run gemma:2b

### 4. Run gemma
ollama run gemma:2b
