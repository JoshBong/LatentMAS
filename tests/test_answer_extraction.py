"""Answer extraction/scoring against REAL judge outputs from the first GPU run.

Every one of these was scored WRONG by the old number-grabbing extractor (it
recorded 1994 / 1945 / 9466 / None) while the model had actually answered
correctly. These lock in the fix.
"""

from utils import answer_hit, extract_answer, token_f1

# (judge response, gold) — condensed from the real console.log
CASES = [
    ("<think>the film ed wood came out in 1994...</think>\n\n"
     "Yes, Scott Derrickson and Ed Wood were both American.", "yes"),
    ("No, the Laleli Mosque and Esma Sultan Mansion are not located in the same neighborhood.", "no"),
    ("The director of the romantic comedy *Big Stone Gap* is **Adriana Trigiani**, "
     "who is based in **Greenwich Village, New York City**.", "greenwich village"),
    ("<think>WINNER debuted in 2014...</think>\n"
     "WINNER was formed by YG Entertainment.\n\n**Answer:** YG Entertainment.", "yg entertainment"),
    ("The person known by his stage name Aladin ... is **Eenasul Fateh**.\n\n"
     "**Answer:** Eenasul Fateh.", "eenasul fateh"),
    ("The **Animorphs** series fits the description.", "animorphs"),
]


def test_real_judge_outputs_now_score_correct():
    for resp, gold in CASES:
        assert answer_hit(resp, gold), f"missed gold={gold!r} in: {resp[:60]!r}"


def test_think_block_is_stripped_not_mined():
    # the CoT mentions 1994; the extracted answer must NOT be that number
    got = extract_answer("<think>...released in 1994...</think>\nYes, both American.")
    assert "1994" not in got
    assert got.lower().startswith("yes")


def test_no_does_not_match_nobody():
    # the padded-substring guard: 'no' must not hit inside 'nobody'
    assert not answer_hit("Nobody knows the answer for certain.", "no")
    assert answer_hit("No, they are different.", "no")


def test_f1_gives_partial_credit():
    assert token_f1("Greenwich Village New York City", "greenwich village") > 0.5
