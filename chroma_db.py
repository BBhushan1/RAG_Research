import chromadb
import hashlib
from embedder import Embedder
from config import VECTOR_DB_PATH

class ChromaVectorDB:
    def __init__(self, embedder=None, persist_path=VECTOR_DB_PATH, collection_name="research_papers"):
        self.embedder = embedder or Embedder()
        try:
            self.client = chromadb.PersistentClient(path=persist_path)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedder
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Chroma client: {str(e)}")

    def generate_id(self, url):
        return hashlib.md5(url.encode()).hexdigest()[:16]

    def add_papers(self, papers):
        if not papers:
            return
        required_keys = {"url", "title", "abstract", "authors", "published"}
        for p in papers:
            if not all(k in p for k in required_keys):
                raise ValueError(f"Paper missing keys: {required_keys - set(p.keys())}")
            
            try:
                p["published"] = int(p["published"])
            except (ValueError, TypeError):
                raise ValueError(f"Invalid year in paper: {p.get('title', 'Unknown')}")

        try:
            ids = [self.generate_id(p['url']) for p in papers]
            texts = [f"{p['title']}. {p['abstract']}" for p in papers]
            metadatas = [
                {
                    "title": p["title"],
                    "abstract": p["abstract"],
                    "authors": ", ".join(p["authors"]),
                    "year": p["published"],
                    "url": p["url"]
                }
                for p in papers
            ]

            self.collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
        except Exception as e:
            raise RuntimeError(f"Failed to add papers to Chroma: {str(e)}")

    def query(self, query_text, n_results=5, year_range=None):
        if not query_text:
            raise ValueError("Query text cannot be empty")
        
        try:
            where = {"year": {"$gte": year_range[0], "$lte": year_range[1]}} if year_range else None
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                include=["metadatas", "distances"]
            )
            
            if not results['documents'][0]:
                return []

            return [
                {
                    "title": meta["title"],
                    "abstract": meta["abstract"],
                    "authors": meta["authors"],
                    "year": meta["year"],
                    "url": meta["url"],
                    "similarity_score": 1 - dist
                }
                for meta, dist in zip(results["metadatas"][0], results["distances"][0])
            ]
        except Exception as e:
            raise RuntimeError(f"Query failed: {str(e)}")