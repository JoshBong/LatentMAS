"""Unit tests for the routing prompt helpers (pure, no model)."""

from prompts_routed import parse_briefs, parse_doc_assignments, worker_doc_slice


def test_parse_briefs_happy_path():
    text = "Worker 1: find the director\nWorker 2: find nationality\nWorker 3: verify"
    briefs, n = parse_briefs(text, 3)
    assert briefs == ["find the director", "find nationality", "verify"]
    assert n == 3                                 # all real, none padded


def test_parse_briefs_reports_underproduction():
    text = ("Here is my plan.\n\n"
            "Worker 1 - locate the film's director.\n"
            "worker 2: identify that person's nationality\n"
            "Some trailing commentary that should be ignored.")
    briefs, n = parse_briefs(text, 3)
    assert len(briefs) == 3                        # padded to n_workers
    assert n == 2                                  # but only 2 were REAL -> caller can detect padding
    assert briefs[0] == "locate the film's director."
    assert briefs[1] == "identify that person's nationality"


def test_parse_briefs_truncates_overproduction():
    text = "\n".join(f"Worker {i}: task {i}" for i in range(1, 7))
    briefs, n = parse_briefs(text, 3)
    assert len(briefs) == 3 and n == 6            # kept 3, saw 6


def test_parse_briefs_strips_docs_tag():
    text = "Worker 1: find the director [docs: 1,3]\nWorker 2: find nationality [docs: 2]"
    briefs, n = parse_briefs(text, 2)
    assert briefs == ["find the director", "find nationality"]     # tag stripped from prose
    assert n == 2


def test_parse_doc_assignments_maps_1based_to_0based():
    text = "Worker 1: a [docs: 1, 3]\nWorker 2: b [docs: 2]\nWorker 3: c [docs: 4]"
    assigns = parse_doc_assignments(text, 3, n_docs=6)
    assert assigns == [[0, 2], [1], [3]]                           # 1-based -> 0-based


def test_parse_doc_assignments_none_when_untagged_and_drops_out_of_range():
    text = "Worker 1: a [docs: 1, 9]\nWorker 2: b"                  # w2 has no tag; 9 is OOB
    assigns = parse_doc_assignments(text, 2, n_docs=3)
    assert assigns == [[0], None]                                  # OOB dropped; untagged -> None


def test_worker_doc_slice_partitions_docs():
    docs = [f"d{i}" for i in range(9)]
    for n in (2, 3, 4):
        parts = [worker_doc_slice(docs, w, n) for w in range(n)]
        assert [d for p in parts for d in p] == docs   # a true partition
    assert worker_doc_slice(None, 0, 3) == []
