from services.extractor.chunk_filter import select_relevant_chunks


def test_select_relevant_chunks_prefers_operational_tables() -> None:
    chunks = [
        {
            "ordinal": 0,
            "heading": "DESTAQUES",
            "content": "VENDAS LÍQUIDAS MRV INCORPORAÇÃO [R$ milhões]\n<!-- image -->",
        },
        {
            "ordinal": 1,
            "heading": "DADOS OPERACIONAIS",
            "content": "| Indicadores Operacionais | 1T25 | 4T24 |\n| VGV (R$ milhoes) | 2.167 | 2.611 |",
        },
    ]

    selected = select_relevant_chunks(chunks, period_label="1T25")

    assert [chunk["ordinal"] for chunk in selected] == [1]
