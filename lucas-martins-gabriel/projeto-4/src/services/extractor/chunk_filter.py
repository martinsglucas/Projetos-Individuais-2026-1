from __future__ import annotations

from typing import Any


IMPORTANT_KEYWORDS = (
    "dados operacionais",
    "indicadores operacionais",
    "land bank",
    "lancamentos",
    "lançamentos",
    "vendas",
    "repasses",
    "producao",
    "produção",
    "geracao de caixa",
    "geração de caixa",
    "vso",
    "vgv",
    "ticket médio",
    "ticket medio",
)

LOW_VALUE_MARKERS = (
    "<!-- image -->",
    "ri.mrv.com.br",
    "disclaimer",
    "informações prospectivas",
    "relacionamento com investidores",
)


def score_chunk(chunk: dict[str, Any], period_label: str | None = None) -> int:
    content = chunk["content"]
    haystack = f"{chunk.get('heading') or ''}\n{content}".lower()

    score = 0
    score += sum(2 for keyword in IMPORTANT_KEYWORDS if keyword in haystack)

    if "|" in content:
        score += 6
    if "var." in haystack:
        score += 2
    if period_label and period_label.lower() in haystack:
        score += 2
    if chunk.get("heading") and "dados" in str(chunk["heading"]).lower():
        score += 3

    image_markers = content.count("<!-- image -->")
    if image_markers and "|" not in content:
        score -= min(image_markers, 5)

    score -= sum(2 for marker in LOW_VALUE_MARKERS if marker in haystack)

    return score


def select_relevant_chunks(
    chunks: list[dict[str, Any]],
    *,
    period_label: str | None = None,
    limit: int = 6,
    min_score: int = 2,
) -> list[dict[str, Any]]:
    scored = [(score_chunk(chunk, period_label=period_label), chunk) for chunk in chunks]
    table_chunks = [(score, chunk) for score, chunk in scored if "|" in chunk["content"] and score >= min_score]

    if table_chunks:
        selected = table_chunks + [
            (score, chunk)
            for score, chunk in scored
            if "|" not in chunk["content"] and score >= 12
        ]
    else:
        selected = [(score, chunk) for score, chunk in scored if score >= min_score]

    if not selected:
        selected = scored

    selected.sort(key=lambda item: (-item[0], item[1]["ordinal"]))
    result = [chunk for _, chunk in selected[:limit]]
    result.sort(key=lambda chunk: chunk["ordinal"])
    return result
