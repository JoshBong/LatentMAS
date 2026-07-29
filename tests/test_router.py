"""Router parsing + self-scoring (pure, no model)."""

from methods.router import parse_router_plan, score_units

WELL_FORMED = """UNITS: Scott Derrickson | Ed Wood
PROPERTY: nationality

WORKER 1
UNIT: Scott Derrickson
OBJECTIVE: Determine Scott Derrickson's nationality.
FORMAT: NATIONALITY: <country>
SOURCES: only passages about Scott Derrickson
BOUNDARIES: do not research Ed Wood; do not compare

WORKER 2
UNIT: Ed Wood
OBJECTIVE: Determine Ed Wood's nationality.
FORMAT: NATIONALITY: <country>
SOURCES: only passages about Ed Wood
BOUNDARIES: do not research Scott Derrickson; do not compare
"""


def test_parse_two_units_and_specs():
    p = parse_router_plan(WELL_FORMED)
    assert p.units == ["Scott Derrickson", "Ed Wood"]
    assert p.property == "nationality"
    assert p.n_workers == 2
    assert p.specs[0].unit == "Scott Derrickson"
    assert "nationality" in p.specs[0].objective.lower()
    assert "ed wood" in p.specs[0].boundaries.lower()          # disjoint boundary present


def test_parse_strips_think_and_prose():
    text = ("<think>the film ed wood was 1994, but that's irrelevant</think>\n"
            "Sure, here is the decomposition:\n\n" + WELL_FORMED)
    p = parse_router_plan(text)
    assert p.units == ["Scott Derrickson", "Ed Wood"]
    assert "1994" not in " ".join(u for u in p.units)


def test_single_unit_bridge_question():
    text = ("UNITS: director of Big Stone Gap\nPROPERTY: city of residence\n\n"
            "WORKER 1\nUNIT: director of Big Stone Gap\n"
            "OBJECTIVE: find the director and where they live\n"
            "FORMAT: CITY: <city>\nSOURCES: passages about Big Stone Gap\n"
            "BOUNDARIES: none\n")
    p = parse_router_plan(text)
    assert p.n_workers == 1                                     # bridge -> not split


def test_score_units_perfect():
    s = score_units(["Scott Derrickson", "Ed Wood"], ["Scott Derrickson", "Ed Wood"])
    assert s["count_match"] and s["precision"] == 1.0 and s["recall"] == 1.0


def test_score_units_role_split_is_penalized():
    # "find entities / find property" style -> a non-entity unit tanks precision/recall
    s = score_units(["Scott Derrickson", "nationality"], ["Scott Derrickson", "Ed Wood"])
    assert s["count_match"]                       # right count, but...
    assert s["precision"] == 0.5 and s["recall"] == 0.5

    s2 = score_units(["the two directors"], ["Scott Derrickson", "Ed Wood"])
    assert not s2["count_match"] and s2["recall"] == 0.0


def test_score_units_matches_title_containment():
    s = score_units(["Scott Derrickson"], ["Scott Derrickson (director)"])
    assert s["recall"] == 1.0
