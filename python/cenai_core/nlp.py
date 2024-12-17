from rapidfuzz import process


def match_text(keyword: str, candidates: list[str]) -> str:
    best_match = process.extractOne(
        keyword, candidates
    )
    return best_match[0]

