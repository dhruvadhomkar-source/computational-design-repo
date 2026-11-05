"""Fast unit tests for the parts that don't need the PDFs or the embedding model."""
from askregs.bm25 import BM25, stem, tokenize
from askregs.evaluate import base_id
from askregs.ingest import MAX_CHARS, PARA_START, clean_lines, split_long


def test_paragraph_numbers_are_detected_but_measurements_are_not():
    assert PARA_START.match("3.30 Flats should be served by more than one common stair").group(1) == "3.30"
    assert PARA_START.match("0.33m2 minimum openable area") is None


def test_running_headers_and_page_numbers_are_removed():
    text = "ONLINE VERSION\n29\nBuilding Regulations 2010 Approved Document B Volume 1, 2019 edition 29\n3.28 The limits on travel"
    assert clean_lines(text) == ["3.28 The limits on travel"]


def test_long_paragraphs_split_into_overlapping_windows_without_losing_text():
    text = " ".join(f"Sentence number {i} about fire doors." for i in range(200))
    parts = split_long(text)
    assert len(parts) > 1
    assert all(len(p) <= MAX_CHARS for p in parts)
    assert parts[0].startswith("Sentence number 0") and parts[-1].endswith("number 199 about fire doors.")


def test_stemming_matches_singular_and_plural():
    assert stem("stairs") == stem("stair") == "stair"
    assert "the" not in tokenize("The stairs")


def test_bm25_ranks_the_matching_document_first():
    bm25 = BM25(["sprinklers in blocks of flats", "handrails on ramps", "door widths in dwellings"])
    scores = bm25.scores("ramp handrail")
    assert scores.index(max(scores)) == 1


def test_window_and_duplicate_suffixes_map_back_to_the_paragraph():
    assert base_id("ADB-V1:3.30#2") == base_id("ADB-V1:3.30~2") == "ADB-V1:3.30"
