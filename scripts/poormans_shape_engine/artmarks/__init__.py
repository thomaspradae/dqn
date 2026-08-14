"""Deterministic mathematical terminal-mark search engine."""

from .families import FAMILY_REGISTRY, get_families
from .models import Candidate, RenderConfig, SearchConfig
from .search import search_marks

__all__ = [
    "Candidate",
    "RenderConfig",
    "SearchConfig",
    "FAMILY_REGISTRY",
    "get_families",
    "search_marks",
]
