import re


def parse_citations(text: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for match in re.finditer(r'=[A-Z]{1,3}\d+', text):
        ref = match.group()
        if ref not in seen:
            seen.add(ref)
            result.append(ref)
    return result
