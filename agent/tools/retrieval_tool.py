from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
load_dotenv()

class KnowledgeBaseRetriever:
    name = "retrieve"
    description = "Search and retrieve internal documents and return relevant chunks. Use for factual or together more infomation about the relevant questions."
    
    def __init__(self, model_name: str, k: int = 3):
        self.k = k
        self.model = SentenceTransformer(model_name)


    def _get_connection(self):
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT"),
            cursor_factory=RealDictCursor
        )
        register_vector(conn)
        return conn

    def search_grouped(self, query: str, user_roles: Optional[List[str]] = None, max_chunks: int = 20) -> Dict[str, List[str]]:
        try:
            embedding = self.model.encode(query).tolist()
        except Exception as e:
            logger.exception("Embedding failed")
            return {"error": [f"FINAL_ANSWER: Failed to process query: {str(e)}"]}

        values = [embedding]
        query_filter = ""
        if user_roles:
            query_filter = "AND permissions && %s"
            values.append(user_roles)

        values1 = values + [self.k]

        sql1 = f"""
            SELECT document_name, file_hash, MAX(1 - (embedding <#> %s::vector)) AS similarity
            FROM documents
            WHERE TRUE {query_filter}
            GROUP BY document_name, file_hash
            ORDER BY similarity DESC
            LIMIT %s
        """

        grouped_chunks = {}

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql1, values1)
                    top_docs = cur.fetchall()

                    #filter by similarity score
                    SIMILARITY_THRESHOLD = 1.35
                    top_docs = [doc for doc in top_docs if doc["similarity"] >= SIMILARITY_THRESHOLD]
                    if not top_docs:
                        return {"error": ["FINAL_ANSWER: No relevant documents found for your query."]}

                    if user_roles and not top_docs:
                        return {"error": ["FINAL_ANSWER: You do not have permission to view the document."]}

                    for doc in top_docs:
                        file_hash = doc["file_hash"]
                        doc_name = doc["document_name"]
                        similarity = doc["similarity"]

                        cur.execute("""
                            SELECT chunk
                            FROM documents
                            WHERE file_hash = %s
                            ORDER BY chunk_index ASC
                            LIMIT %s
                        """, (file_hash, max_chunks))

                        chunks = cur.fetchall()
                        grouped_chunks[f"{doc_name} (score: {similarity:.4f})"] = [row["chunk"] for row in chunks]

        except Exception as e:
            logger.exception("Database query failed")
            return {"error": [f"FINAL_ANSWER: Failed to retrieve documents: {str(e)}"]}

        return grouped_chunks

'''
if __name__ == "__main__":
    retriever = KnowledgeBaseRetriever(model_name="Qwen/Qwen3-Embedding-0.6B", k=3)
    
    query = "Harry potter"
    roles = []  # or None if unrestricted

    result = retriever.search_grouped(query, user_roles=roles)
    
    for doc_name, chunks in result.items():
        print(f"\n📄 {doc_name}")
        for i, chunk in enumerate(chunks):
            print(f"  {i+1}. {chunk[:100]}...")

'''