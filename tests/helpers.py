"""Test helpers: build real PDF and DOCX files in memory."""

from __future__ import annotations

import io


def make_pdf(pages: list[str]) -> bytes:
    """Write a minimal but valid text-layer PDF (Helvetica, one text line per input line)."""
    objects: list[bytes] = []
    n_pages = len(pages)
    page_ids = [3 + 2 * i for i in range(n_pages)]
    font_id = 3 + 2 * n_pages
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i, text in enumerate(pages):
        lines = []
        for j, line in enumerate(text.split("\n")):
            safe = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            lines.append(f"BT /F1 12 Tf 72 {720 - 16 * j} Td ({safe}) Tj ET")
        stream = "\n".join(lines).encode()
        content_id = page_ids[i] + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for num, body in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{num} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    )
    return out.getvalue()


def make_docx(paragraphs: list[tuple[str, str]]) -> bytes:
    """paragraphs: (style, text) pairs, e.g. ("Heading 1", "Intro"), ("Normal", "Body")."""
    import docx

    document = docx.Document()
    for style, text in paragraphs:
        document.add_paragraph(text, style=style)
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()
