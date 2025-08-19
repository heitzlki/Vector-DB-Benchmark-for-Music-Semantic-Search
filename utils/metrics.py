from typing import List, Dict, Any
import re


def normalize_tags(raw: str) -> List[str]:
    if not isinstance(raw, str):
        return []
    # Try to parse lists like "['aggressive', 'calm']" or comma strings
    inside = raw.strip()
    # Remove brackets and quotes
    inside = inside.replace("[", "").replace("]", "")
    parts = re.split(r"[,'\"]+", inside)
    tags = [p.strip().lower() for p in parts if p.strip()]
    return tags


def relevance_hit(payload: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    # Heuristic: match if any expected tag is in seeds OR expected genre is in genre
    seeds = normalize_tags(payload.get("seeds", ""))
    genre = str(payload.get("genre", "")).lower()
    exp_tags = [t.lower() for t in expected.get("tags", [])]
    exp_genres = [g.lower() for g in expected.get("genres", [])]
    tag_match = any(t in seeds for t in exp_tags) if exp_tags else False
    genre_match = any(g in genre for g in exp_genres) if exp_genres else False
    return tag_match or genre_match


def hits_at_k(results_payloads: List[Dict[str, Any]], expected: Dict[str, Any]) -> int:
    # If expected is a set, treat as set of row_ids
    if isinstance(expected, set):
        return sum(1 for p in results_payloads if p.get("row_id") in expected)
    # Otherwise, fallback to old tag/genre logic
    return sum(1 for p in results_payloads if relevance_hit(p, expected))
