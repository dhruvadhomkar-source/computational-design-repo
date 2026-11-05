"""Turn Approved Document PDFs into citable chunks.

Approved Documents are written as numbered paragraphs ("3.30 Flats should be served by...").
A paragraph is the unit people cite, so it is also our chunk: every chunk keeps the paragraph
number, the section it sits in and the printed page it starts on. Long paragraphs are split
into overlapping windows that keep the same paragraph number.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pypdf

MAX_CHARS = 1500      # longer paragraphs are split into windows of this size...
OVERLAP_CHARS = 200   # ...overlapping by this much, so no sentence is lost at a cut

# A numbered paragraph starts a line: "3.30 Flats should..." or "1.5 A large dwellinghouse...".
# Requiring a capital letter after the number avoids matching measurements like "0.33m2".
PARA_START = re.compile(r"^(\d{1,2}\.\d{1,3})\s+(?=[A-Z‘'\"(])")
SECTION = re.compile(r"^(Section \d+[A-Z]?:.*|Appendix [A-Z]:.*)$")

# Lines that repeat on every page and carry no content.
NOISE = [
    re.compile(r"^ONLINE VERSION$"),
    re.compile(r"^Building Regulations 2010\s+Approved Document .*edition\s*\d*$"),
    re.compile(r"^Approved Document M Volume 2 Access to and use of buildings other than dwellings$"),
    re.compile(r"^\d{1,3}$"),                    # bare page numbers
    re.compile(r"^(B\d|M\d|M4\(\d\))$"),          # requirement tags in the margin
]


@dataclass
class Chunk:
    id: str          # e.g. "ADB-V1:3.30" or "ADB-V1:3.30#2" for the 2nd window
    doc_id: str
    para: str        # paragraph number as printed, "" for text before the first paragraph
    section: str
    page: str        # printed page label, what a reader sees in the document
    pdf_page: int    # 1-based PDF page, for opening the file at the right place
    text: str


def clean_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = " ".join(raw.split())  # collapse runs of spaces
        if line and not any(p.match(line) for p in NOISE):
            lines.append(line)
    return lines


def split_long(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    parts, start = [], 0
    while start < len(text):
        end = min(start + MAX_CHARS, len(text))
        if end < len(text):  # cut at the last sentence or word boundary inside the window
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            end = cut + 1 if cut > start + MAX_CHARS // 2 else end
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - OVERLAP_CHARS
    return parts


def chunk_document(pdf_path: Path, doc_id: str) -> list[Chunk]:
    reader = pypdf.PdfReader(str(pdf_path))
    labels = reader.page_labels

    chunks: list[Chunk] = []
    section = ""
    current = None  # (para, section, page_label, pdf_page, [lines])

    def flush():
        if current is None:
            return
        para, sec, label, pdf_page, lines = current
        text = " ".join(lines).strip()
        if len(text) < 40:  # headings and stray fragments are not worth indexing alone
            return
        windows = split_long(text)
        for n, window in enumerate(windows, start=1):
            suffix = f"#{n}" if len(windows) > 1 else ""
            key = para or f"p{label}"
            chunks.append(Chunk(f"{doc_id}:{key}{suffix}", doc_id, para, sec, label, pdf_page, window))

    for index, page in enumerate(reader.pages):
        label = labels[index]
        for line in clean_lines(page.extract_text() or ""):
            if SECTION.match(line):
                section = line
            match = PARA_START.match(line)
            if match:
                flush()
                current = (match.group(1), section, label, index + 1, [line])
            elif current is None:
                current = ("", section, label, index + 1, [line])
            else:
                current[4].append(line)
    flush()
    return dedupe_ids(chunks)


def dedupe_ids(chunks: list[Chunk]) -> list[Chunk]:
    """Paragraph numbers can repeat (e.g. in appendices); keep every id unique."""
    seen: dict[str, int] = {}
    for chunk in chunks:
        if chunk.id in seen:
            seen[chunk.id] += 1
            chunk.id = f"{chunk.id}~{seen[chunk.id]}"
        else:
            seen[chunk.id] = 1
    return chunks


def ingest(sources_file: Path, raw_dir: Path, out_file: Path) -> list[Chunk]:
    sources = json.loads(sources_file.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for source in sources:
        chunks += chunk_document(raw_dir / source["file"], source["id"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return chunks


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
