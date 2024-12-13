import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_elasticsearch import ElasticsearchStore
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.llms import OllamaLLM


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


class langChainAgent:
    """
    Docs:
    1. https://python.langchain.com/docs/tutorials/rag/
    2. https://python.langchain.com/docs/integrations/llms/ollama/
    3. https://python.langchain.com/docs/tutorials/retrievers/
    """

    def __init__(self):
        pass

    def start_agent(self, pdf_path):
        """
        Instantiation processes PDF, creates embeddings, 
        vector store and starts an LLM.
        """
        # Load and chunk contents of the blog
        all_splits = self._load_and_chunk_pdf(pdf_path)

        # Index chunks
        self.vector_store = self._create_es_vector_score(all_splits)

        # Define prompt for question-answering
        self.prompt = hub.pull("rlm/rag-prompt")

        # init ollama LLM
        self.llm = OllamaLLM(model="gemma:2b")

    def _load_and_chunk_pdf(self, pdf_path):
        """
        Load pdf from local directory.
        """
        # init loader
        loader = PyPDFLoader(pdf_path)
        # load docs
        docs = loader.load()
        # chunk data with text oberlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)

        return all_splits

    def _create_ollama_embeddings(self):
        """
        Use ollama to create embeddings.
        """
        # init
        embeddings = OllamaEmbeddings(model="gemma:2b")
        return embeddings

    def _create_es_vector_score(self, all_splits):
        """
        Create elastic search vector store
        """
        # use ollamba to generate embeddings
        embeddings = self._create_ollama_embeddings()

        try:
            vector_store = ElasticsearchStore(
                index_name="langchain-demo", 
                embedding=embeddings, 
                es_url="http://localhost:9200"
            )
        except Exception as e:
            print(f"Elasticsearch error occurred: {e}")

        # add pdf splits to vector store
        vector_store.add_documents(documents=all_splits)

        return vector_store

    # Define application steps
    def retrieve(self, state: State):
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}


    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}

    def compile_graph(self):
        """
        """
        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        return graph