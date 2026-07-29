"""Unit tests for the routing prompt helpers (pure, no model)."""

from prompts_routed import parse_briefs, worker_doc_slice


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


def test_worker_doc_slice_partitions_docs():
    docs = [f"d{i}" for i in range(9)]
    for n in (2, 3, 4):
        parts = [worker_doc_slice(docs, w, n) for w in range(n)]
        assert [d for p in parts for d in p] == docs   # a true partition
    assert worker_doc_slice(None, 0, 3) == []
