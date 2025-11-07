from .ordering import Box
from typing import List, Dict

def render_markdown_html(boxes: List[Box], page_tables: Dict[int, list]) -> tuple[str, str]:
    """Render extracted Boxes and Tables into markdown/plaintext and HTML."""
    plain_lines = []
    html_lines = ['<html><head><style>',
                  'body { font-family: Arial; line-height: 1.6; }',
                  'h2 { margin-top: 40px; }',
                  'table { border-collapse: collapse; width: 100%; margin: 10px 0; }',
                  'td, th { border: 1px solid #ccc; padding: 6px; text-align: left; }',
                  '</style></head><body>']

    prev_page = -1
    for b in boxes:
        if b.page != prev_page:
            # New page header
            plain_lines.append(f"\n\n=== Page {b.page} ===\n")
            html_lines.append(f"<h2>Page {b.page}</h2>")
            prev_page = b.page

        # Style by kind
        if b.kind == "heading":
            plain_lines.append(f"\n# {b.text}\n")
            html_lines.append(f"<h3>{b.text}</h3>")
        elif b.kind == "table":
            continue  # handled later
        elif b.kind == "para":
            plain_lines.append(b.text)
            html_lines.append(f"<p>{b.text}</p>")
        else:
            plain_lines.append(b.text)
            html_lines.append(f"<div>{b.text}</div>")

    # Render tables
    for i, table in enumerate(page_tables):
        plain_lines.append(f"\n[Table {i + 1}]")
        html_lines.append(f"<h4>Table {i + 1}</h4><table>")
        for row in table["headers"] + table["rows"]:
            html_lines.append("<tr>")
            for cell in row:
                text = cell.get("text", "")
                html_lines.append(f"<td>{text}</td>")
            html_lines.append("</tr>")
        html_lines.append("</table>")

    html_lines.append("</body></html>")
    return "\n".join(plain_lines), "\n".join(html_lines)
