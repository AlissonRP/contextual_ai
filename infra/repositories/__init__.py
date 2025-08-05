"""
Expose a single ChromaRepository instance that the whole app can share.
`vector_repo` is kept as an alias for backward-compatibility.
"""

from infra.repositories.chroma_repository import ChromaRepository


chroma_repo = ChromaRepository()

# alias usado em partes antigas do código
vector_repo = chroma_repo
