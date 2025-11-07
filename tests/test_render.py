from src.render import render_markdown_html
from src.ordering import Box

def test_render_basic_paragraphs_and_table():
    boxes = [
        Box(page=0, x0=0.1, y0=0.1, x1=0.2, y1=0.2, text="Title", kind="heading"),
        Box(page=0, x0=0.1, y0=0.3, x1=0.2, y1=0.4, text="Para line 1.", kind="para")
    ]
    page_tables = [{
        "page": 0,
        "rows": [["H1", "H2"], ["v1", "v2"]],
        "n_header_rows": 1
    }]
    md, html = render_markdown_html(boxes, page_tables)
    assert "# Title" in md or "Title" in md
    assert "Para line 1." in md
    assert "| H1 | H2 |" in md
