import os
from linguo.vector_db import ChromaDB
from linguo.types.documents import Documents
from linguo.aws import SageMakerInferenceClient
from linguo.models.rerankers import AIPReranker
from linguo.models.embedders import OpenAIEmbedder


class Retriever:

    def __init__(self) -> None:
        self.embedder = OpenAIEmbedder(
            model_name="text-embedding-3-large-cde-aia",
            azure_endpoint=os.getenv("OPENAI_API_BASE_2"),
            api_key=os.getenv("OPENAI_API_KEY_2"),
            api_version=os.getenv("OPENAI_API_VERSION_2")
        )
        self.vector_db = ChromaDB(
            db_bucket_name="ai-prototype",
            db_bucket_key="question_solver/agents-experiment/chroma/"
        )

        self.vector_db_k = 5

        self.reranker = AIPReranker(
            model_endpoint="bancolombia-reranker-ep",
            sm_client=SageMakerInferenceClient()
        )
        self.reranker_threshold = 0.55
        self.reranker_batch_size = 10
        self.reranker_k = 5

    def retrieve(self, query: str, table_name: str) -> Documents:
        query_embeddings = self.embedder.embed([query], 1)
        documents = self.vector_db.vector_search_docs(
            query_embeddings=query_embeddings[0],
            table_name=table_name,
            k=self.vector_db_k
        )

        reranked_documents = self.reranker.rerank(
            query=query,
            documents=documents,
            batch_size=self.reranker_batch_size
        )

        filtered_documents = [doc
                              for doc in reranked_documents.docs
                              if doc.score > self.reranker_threshold]
        filtered_documents = filtered_documents[:self.reranker_k]
        filtered_documents = Documents(documents=filtered_documents)

        return filtered_documents


if __name__ == "__main__":
    table_names = [
        "comex",
        "gestion_humana"
    ]
    retriever = Retriever()
    question = "Que debo tener en cuenta para solicitar" + \
               " un segundo credito de vivienda?"

    docs = retriever.retrieve(query=question,
                              table_name=table_names[1])
    print(docs.get_texts_as_str(token=f"\n\n\n{100*'#'}\n"))
