# tests/test_tables.py

from src.tables import tables_for_doc
from types import SimpleNamespace as NS

def test_reconstruct_simple_table(simple_table_doc):
    tables = tables_for_doc(simple_table_doc)
    assert isinstance(tables, list)
    assert len(tables) == 1

    t = tables[0]
    assert t["n_header_rows"] == 1
    assert len(t["headers"]) == 1
    assert t["headers"][0][0]["text"] == "H1"
    assert t["headers"][0][1]["text"] == "H2"
    assert t["rows"] == []  # No body rows in test doc
