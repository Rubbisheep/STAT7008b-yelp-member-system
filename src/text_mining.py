import re
from collections import Counter
from typing import Dict, Iterable, List


POSITIVE = {"great", "good", "love", "smooth", "comfortable", "awesome"}
NEGATIVE = {"bad", "poor", "inaccurate", "expensive", "boring", "annoying"}


def _tokenize(text: str) -> Iterable[str]:
    for token in re.split(r"[\\s,.;!?]", text):
        token = token.strip()
        if token:
            yield token


def extract_keywords(text: str, top_k: int = 5) -> List[str]:
    if not text:
        return []
    tokens = [t.lower() for t in _tokenize(text)]
    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(top_k)]


def predict_mbti(text: str) -> str:
    text_lower = text.lower()
    axes = []
    axes.append("E" if ("social" in text_lower or "friends" in text_lower) else "I")
    axes.append("N" if ("imagine" in text_lower or "idea" in text_lower) else "S")
    axes.append("T" if ("logic" in text_lower or "analyze" in text_lower) else "F")
    axes.append("J" if ("plan" in text_lower or "efficient" in text_lower or "deadline" in text_lower) else "P")
    return "".join(axes)


def classify_user_segment(features: Dict[str, float]) -> str:
    play_count = features.get("play_count", 0)
    duration = features.get("total_duration", 0)
    avg_rating = features.get("avg_rating", 0)
    if play_count > 150 or duration > 20000:
        return "heavy_user"
    if avg_rating < 3:
        return "price_sensitive"
    if play_count < 10:
        return "occasional_user"
    return "steady_user"
