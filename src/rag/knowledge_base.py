"""
RAG knowledge base.

Loads curated genre guides and music articles from data/knowledge/, chunks and
embeds them into a ChromaDB vector store, and exposes a retrieval interface for
fetching context-relevant passages to ground agent recommendations.
"""
