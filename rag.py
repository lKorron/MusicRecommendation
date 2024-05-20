from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

from config import config

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = None
Settings.chunk_size = config["chunk_size"]
Settings.chunk_overlap = config["chunk_overlap"]

documents = SimpleDirectoryReader("rag_data").load_data()
index = VectorStoreIndex.from_documents(documents)

def get_context(query, top_k = 3):
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=config["similarity_cutoff"])],
    )

    response = query_engine.query(query)

    context = "Context:\n"
    for i in range(top_k):
        context = context + response.source_nodes[i].text + "\n\n"

    return context

