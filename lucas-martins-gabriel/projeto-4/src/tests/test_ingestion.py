from services.ingestion.poll_sources import extract_mziq_config


def test_extract_mziq_config_finds_preview_category() -> None:
    html = """
    <script>
    var fmId = 'abc-123';
    categories.push({
      title: 'Release de Resultados',
      internal_name: 'release'
    })
    categories.push({
      title: 'Prévia Operacional',
      internal_name: 'previa_operacional'
    })
    </script>
    """

    fm_id, categories = extract_mziq_config(html)

    assert fm_id == "abc-123"
    assert categories == ["previa_operacional"]
