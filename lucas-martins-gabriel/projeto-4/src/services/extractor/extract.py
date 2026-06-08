from docling.document_converter import DocumentConverter

converter = DocumentConverter()

result = converter.convert("../../data/raw/mrv_1t25.pdf")

markdown = result.document.export_to_markdown()

print(markdown[:5000])