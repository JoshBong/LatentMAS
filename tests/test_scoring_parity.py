"""Every arm must score identically. This is the test that was missing.

The bug this locks down: baseline.py and text_mas.py had no free-form branch, so
on hotpotqa they fell through to "grab the last number in the string" and scored
~0 no matter what the model said -- while latent_mas used whole-word recall over
the entire raw response (inflated) and routed_mas used extraction-based EM
(correct). Three arms, three metrics, one verdict table in experiments/analyze.py.

Run: python -m pytest tests/test_scoring_parity.py -q     (or: python tests/test_scoring_parity.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (  # noqa: E402
    CODE_TASKS,
    NUMERIC_TASKS,
    extract_prediction,
    numeric_eq,
    score_prediction,
)

# A realistic hotpotqa judge response: <think> block, then prose leading with the
# answer. Note the digits in the CoT -- the old baseline path returned "1978".
JUDGE_HOTPOT = """<think>
Worker 1 says the film was directed by Ang Lee. Worker 2 says Ang Lee was born in
1954 in Pingtung. The other candidate, Wong Kar-wai, was born in 1958.
</think>
The answer is Ang Lee."""

JUDGE_HOTPOT_RAMBLE = """<think>reasoning</think>
Possible directors include Wong Kar-wai, Ang Lee, and Hou Hsiao-hsien. I cannot
determine which one from the evidence provided."""

JUDGE_GSM = "<think>3 boxes at 6 each</think>\nThe answer is \\boxed{18}."


def test_freeform_extraction_is_not_a_number_grab():
    """The old baseline path returned the last number in the string."""
    pred = extract_prediction(JUDGE_HOTPOT, "hotpotqa")
    assert pred == "Ang Lee", pred
    assert not pred.isdigit()


def test_all_arms_agree_on_a_correct_freeform_answer():
    pred, ok, f1 = score_prediction(JUDGE_HOTPOT, "Ang Lee", "hotpotqa")
    assert (pred, ok) == ("Ang Lee", True)
    assert f1 == 1.0


def test_rambling_answer_is_not_credited():
    """latent_mas's old answer_hit() gave this credit: 'ang lee' appears as whole
    words in the raw response. Extraction-based EM must not."""
    pred, ok, f1 = score_prediction(JUDGE_HOTPOT_RAMBLE, "Ang Lee", "hotpotqa")
    assert ok is False, f"rambling non-answer was credited (pred={pred!r})"
    assert f1 < 1.0


def test_squad_normalisation_applies():
    for variant in ("Ang Lee.", "ang lee", "  Ang  Lee  ", "The Ang Lee"):
        _, ok, _ = score_prediction(f"The answer is {variant}", "Ang Lee", "hotpotqa")
        assert ok, variant


def test_numeric_tasks_still_work_and_are_now_type_tolerant():
    pred, ok, _ = score_prediction(JUDGE_GSM, "18", "gsm8k")
    assert (pred, ok) == ("18", True)
    # the old `normalize_answer(pred) == gold` string compare failed these
    assert numeric_eq("18.0", "18")
    assert numeric_eq("1,000", "1000")
    assert not numeric_eq("19", "18")


def test_empty_and_missing_gold_are_safe():
    assert score_prediction("", "Ang Lee", "hotpotqa")[1] is False
    assert score_prediction(JUDGE_HOTPOT, "", "hotpotqa")[1] is False
    assert score_prediction(JUDGE_HOTPOT, None, "hotpotqa")[1] is False


def test_methods_share_one_implementation():
    """routed_mas._score must be the same code path as everyone else."""
    import argparse
    from methods.routed_mas import RoutedMASMethod

    m = RoutedMASMethod.__new__(RoutedMASMethod)   # no model load
    m.task = "hotpotqa"
    assert m._score(JUDGE_HOTPOT, "Ang Lee") == score_prediction(
        JUDGE_HOTPOT, "Ang Lee", "hotpotqa"
    )[:2]
    assert m._extract(JUDGE_HOTPOT) == extract_prediction(JUDGE_HOTPOT, "hotpotqa")
    del argparse


def test_task_sets_are_disjoint():
    assert not (NUMERIC_TASKS & CODE_TASKS)
    assert "hotpotqa" not in NUMERIC_TASKS and "hotpotqa" not in CODE_TASKS


def test_the_regression_that_started_this():
    """Direct reproduction of the old baseline.py behaviour vs the new one."""
    from utils import extract_gsm8k_answer, normalize_answer

    old_pred = normalize_answer(extract_gsm8k_answer(JUDGE_HOTPOT))   # legacy path
    old_ok = (old_pred == "Ang Lee")
    new_pred, new_ok, _ = score_prediction(JUDGE_HOTPOT, "Ang Lee", "hotpotqa")

    assert old_pred == "1958" and old_ok is False   # <- the artifact
    assert new_pred == "Ang Lee" and new_ok is True


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                fails += 1
                print(f"  FAIL  {name}: {e}")
    print("\nall green" if not fails else f"\n{fails} failing")
    sys.exit(1 if fails else 0)
