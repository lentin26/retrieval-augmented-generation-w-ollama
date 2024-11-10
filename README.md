
### 1. Pull image
docker pull ollama/ollama

### 2. Run container in detached mode
docker run -d ollama/ollama

### 3. Exec into container
docker exec -it ollama ollama run gemma:2b

### 4. Run gemma
ollama run gemma:2b
# retrieval-augmented-generation-w-ollama
