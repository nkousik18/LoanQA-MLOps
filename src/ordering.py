# src/ordering.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Iterable, Any

# Keep annotations import-safe for tests that don't install DocAI
try:
    from google.cloud import documentai_v1 as documentai  # type: ignore
except Exception:
    documentai = Any  # type: ignore


@dataclass
class Box:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    kind: str  # "para", "header", "footer", etc.


def get_text(doc: Any, text_anchor: Any) -> str:
    """Resolve Document AI textAnchor into text. Supports snake/camel case stubs."""
    if not text_anchor:
        return ""
    segments = (
        getattr(text_anchor, "text_segments", None)
        or getattr(text_anchor, "textSegments", None)
        or []
    )
    out = []
    for seg in segments:
        start = int(getattr(seg, "start_index", getattr(seg, "startIndex", 0) or 0))
        end = int(getattr(seg, "end_index", getattr(seg, "endIndex", 0) or 0))
        out.append(doc.text[start:end])
    return "".join(out)


def _norm_bbox(poly: Any) -> Tuple[float, float, float, float]:
    """Convert a bounding_poly to normalized (x0, y0, x1, y1)."""
    if not poly:
        return (0.0, 0.0, 1.0, 1.0)

    verts = (
        getattr(poly, "normalized_vertices", None)
        or getattr(poly, "normalizedVertices", None)
        or getattr(poly, "vertices", None)
        or []
    )
    xs = [float(getattr(v, "x", 0.0)) for v in verts] or [0.0, 1.0]
    ys = [float(getattr(v, "y", 0.0)) for v in verts] or [0.0, 1.0]
    return (min(xs), min(ys), max(xs), max(ys))


def _columnize(items: List[Box], min_col_gap: float = 0.08) -> List[Box]:
    """
    Simple multi-column ordering:
      1) cluster by x-mid using gap threshold
      2) order columns left→right; inside each, top→bottom
    """
    if not items:
        return []

    items = sorted(items, key=lambda b: (b.x0 + b.x1) / 2.0)

    columns: List[List[Box]] = [[items[0]]]
    for b in items[1:]:
        xmid = (b.x0 + b.x1) / 2.0
        placed = False
        for col in columns:
            cx = (col[-1].x0 + col[-1].x1) / 2.0
            if abs(xmid - cx) <= min_col_gap:
                col.append(b)
                placed = True
                break
        if not placed:
            columns.append([b])

    def col_x(col: List[Box]) -> float:
        xs = [ (p.x0 + p.x1)/2.0 for p in col ]
        xs.sort()
        return xs[len(xs)//2]

    columns.sort(key=col_x)
    for col in columns:
        col.sort(key=lambda b: (b.y0, b.x0))

    ordered: List[Box] = []
    for col in columns:
        ordered.extend(col)
    return ordered


def _iter_paragraph_nodes(page: Any) -> Iterable[Any]:
    """Yield paragraph-like nodes from page.paragraphs or page.blocks[*]."""
    if getattr(page, "paragraphs", None):
        for p in page.paragraphs:
            yield p
        return

    for b in getattr(page, "blocks", None) or []:
        t = (getattr(b, "type", "") or "").lower()
        if t in {"paragraph", "para"}:
            yield b


def _x_overlap_ratio(a: Box, b: Box) -> float:
    """Horizontal overlap ratio of two boxes (0..1)."""
    left = max(a.x0, b.x0)
    right = min(a.x1, b.x1)
    width = max(0.0, right - left)
    denom = max(a.x1 - a.x0, b.x1 - b.x0, 1e-6)
    return width / denom


def merge_hyphenated(boxes: List[Box], gap: float = 0.035, min_x_overlap: float = 0.35) -> List[Box]:
    """
    Merge lines split with a trailing hyphen where the next line is very close vertically
    and roughly in the same column (x-overlap).
    """
    if not boxes:
        return boxes

    merged: List[Box] = []
    i = 0
    while i < len(boxes):
        cur = boxes[i]
        if (
            i + 1 < len(boxes)
            and cur.text.rstrip().endswith("-")
        ):
            nxt = boxes[i + 1]
            same_page = cur.page == nxt.page
            vertical_close = (nxt.y0 - cur.y1) >= -1e-6 and (nxt.y0 - cur.y1) <= gap
            same_col = _x_overlap_ratio(cur, nxt) >= min_x_overlap
            if same_page and vertical_close and same_col:
                # join words: drop trailing hyphen and glue next text
                new_text = cur.text.rstrip().rstrip("-") + nxt.text.lstrip()
                cur = Box(
                    page=cur.page,
                    x0=min(cur.x0, nxt.x0),
                    y0=min(cur.y0, nxt.y0),
                    x1=max(cur.x1, nxt.x1),
                    y1=max(cur.y1, nxt.y1),
                    text=new_text,
                    kind=cur.kind,
                )
                i += 2
                merged.append(cur)
                continue
        merged.append(cur)
        i += 1
    return merged


def reconstruct_boxes(doc: documentai.Document) -> List[Box]:
    """
    Build ordered list of paragraph-like boxes for all pages.
    Works with real DocAI objects and test stubs (duck-typed).
    """
    out: List[Box] = []

    for i, page in enumerate(getattr(doc, "pages", []) or [], start=1):
        # Optional headers/footers
        for hf_kind in ("header", "footer"):
            hf_list = getattr(page, f"{hf_kind}_footer", None) or getattr(page, hf_kind, None)
            if hf_list:
                for hf in hf_list:
                    layout = getattr(hf, "layout", None)
                    if not layout:
                        continue
                    out.append(
                        Box(i, *_norm_bbox(layout.bounding_poly), get_text(doc, layout.text_anchor), hf_kind)
                    )

        paras = list(_iter_paragraph_nodes(page))

        # Deduplicate by text anchor span tuple
        seen: set = set()
        pboxes: List[Box] = []
        for node in paras:
            layout = getattr(node, "layout", None)
            if not layout:
                continue
            ta = layout.text_anchor
            spans = (
                getattr(ta, "text_segments", None)
                or getattr(ta, "textSegments", None)
                or []
            )
            key = tuple(
                (int(getattr(s, "start_index", getattr(s, "startIndex", 0) or 0)),
                 int(getattr(s, "end_index", getattr(s, "endIndex", 0) or 0)))
                for s in spans
            )
            if key in seen:
                continue
            seen.add(key)

            text = get_text(doc, ta).strip()
            if not text:
                continue

            x0, y0, x1, y1 = _norm_bbox(layout.bounding_poly)
            pboxes.append(Box(i, x0, y0, x1, y1, text, "para"))

        # Column-aware ordering + soft line joining for hyphenation
        ordered = _columnize(pboxes)
        ordered = merge_hyphenated(ordered)
        out.extend(ordered)

    return out
