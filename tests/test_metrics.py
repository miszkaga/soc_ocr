import pytest

from eval.metrics import (GtWord, PredWord, accept_table, aggregate, categories, coverage,
                          evaluate_image, expected_calibration_error)
from ocr.config import load_config

CFG = load_config().eval


def g(text, box, **kw):
    return GtWord(text=text, bbox=box, **kw)


def p(text, box, conf=0.95):
    return PredWord(text=text, bbox=box, conf=conf)


def cats(text, **kw):
    return categories(GtWord(text=text, bbox=(0, 0, 1, 1), **kw)) - {"all"}


@pytest.mark.parametrize("text,expected", [
    ("j.kowalski@example.com", {"email", "hard"}),
    ("10.0.0.1", {"ip", "hard"}),
    ("10.100.0.0/16", {"ip", "hard"}),
    ("192.168.1.1:8080", {"ip", "hard"}),
    ("fe80::1ff:fe23:4567:890a", {"ip", "hard"}),
    ("privatelink.blob.core.windows.net", {"host", "hard"}),
    ("https://portal.azure.com", {"host", "hard"}),
    ("250", {"number", "hard"}),
    ("2024-10-20", {"number", "hard"}),
    ("service1:80", {"id", "hard"}),
    ("Stgworkload00", {"id", "hard"}),
    ("Zażółć", {"polish", "hard"}),
    ("Ingress,", set()),
    ("Firewall", set()),
])
def test_categories(text, expected):
    assert cats(text) == expected


def test_rotated_words_get_their_own_category():
    assert cats("Peered", rotated=True) == {"rotated"}


def test_perfect_reading():
    gt = [g("Azure", (0, 0, 50, 20)), g("10.0.0.1", (60, 0, 140, 20))]
    pred = [p("Azure", (1, 4, 49, 16)), p("10.0.0.1", (61, 4, 139, 16))]  # tight boxes inside line boxes
    r = aggregate([evaluate_image(gt, pred, CFG)], CFG)
    a = r["categories"]["all"]
    assert (a["words"], a["detection_recall"], a["cer"], a["wer"]) == (2, 1.0, 0.0, 0.0)
    assert r["false_positives"] == 0 and r["errors"] == []
    assert r["categories"]["ip"]["cer"] == 0.0


def test_confusable_substitution_costs_one_edit():
    r = aggregate([evaluate_image([g("10.0.0.1", (0, 0, 80, 20))], [p("l0.0.0.1", (0, 2, 80, 18))], CFG)], CFG)
    a = r["categories"]["all"]
    assert a["cer"] == pytest.approx(1 / 8) and a["wer"] == 1.0 and a["detection_recall"] == 1.0
    assert r["errors"][0] | {} == {"gt": "10.0.0.1", "pred": "l0.0.0.1", "conf": 0.95, "detected": True,
                                    "categories": ["hard", "ip"]}


def test_missed_word_costs_all_its_characters():
    gt = [g("Azure", (0, 0, 50, 20)), g("DNS", (60, 0, 90, 20))]
    r = aggregate([evaluate_image(gt, [p("Azure", (0, 2, 50, 18))], CFG)], CFG)
    a = r["categories"]["all"]
    assert a["detection_recall"] == 0.5
    assert a["cer"] == pytest.approx(3 / 8)
    assert r["errors"][0]["pred"] is None and not r["errors"][0]["detected"]


def test_hallucinated_word_is_a_false_positive_and_scored_wrong():
    res = evaluate_image([g("Azure", (0, 0, 50, 20))], [p("Azure", (0, 2, 50, 18)), p("III", (200, 0, 230, 20), 0.6)], CFG)
    assert res.false_positives == 1
    assert (0.6, False, 3, 0, None) in res.scored


def test_ignored_words_are_not_scored_and_not_false_positives():
    gt = [g("Azure", (0, 0, 50, 20)), g("Poc", (100, 0, 130, 20), ignore=True)]
    res = evaluate_image(gt, [p("Azure", (0, 2, 50, 18)), p("Poc", (100, 2, 130, 18))], CFG)
    assert res.counts["all"].words == 1 and res.false_positives == 0


def test_split_word_is_detected_and_second_piece_is_not_a_false_positive():
    res = evaluate_image([g("10.0.0.1", (0, 0, 80, 20))], [p("10.0.0.", (0, 2, 68, 18)), p("1", (70, 2, 80, 18))], CFG)
    assert res.counts["all"].detected == 1
    assert res.false_positives == 0
    # The two pieces are read together: "10.0.0." + "1" is the word.
    assert res.counts["all"].edits == 0 and res.counts["all"].exact == 1


def test_coverage_along_long_axis_for_vertical_words():
    vertical = (100, 0, 120, 200)  # rotated label
    assert coverage(vertical, [(102, 0, 118, 190)]) == pytest.approx(0.95)
    assert coverage(vertical, [(0, 0, 90, 200)]) == 0.0  # beside it, not on it


def test_coverage_merges_overlapping_predictions():
    assert coverage((0, 0, 100, 20), [(0, 0, 60, 20), (40, 0, 80, 20)]) == pytest.approx(0.8)


def test_expected_calibration_error():
    assert expected_calibration_error([(0.9, True, 0, 1, None), (0.9, False, 1, 1, None)], 10) == pytest.approx(0.4)
    assert expected_calibration_error([(1.0, True, 0, 1, None)], 10) == 0.0
    assert expected_calibration_error([], 10) is None


def test_accept_table_trades_review_for_quality():
    scored = [(0.99, True, 0, 5, None), (0.98, True, 0, 4, None), (0.6, False, 2, 5, None), (0.5, False, 3, 0, None)]
    rows = {r["threshold"]: r for r in accept_table(scored, gt_words=4, thresholds=(0.0, 0.9))}
    assert rows[0.0]["accepted_words"] == 3 and rows[0.0]["accepted_hallucinations"] == 1
    assert rows[0.0]["accepted_cer"] == pytest.approx(5 / 14)
    assert rows[0.9]["accepted_cer"] == 0.0 and rows[0.9]["accepted_word_error"] == 0.0
    assert rows[0.9]["review_rate"] == pytest.approx(0.5)  # 2 of 4 words accepted


def test_aggregate_sums_images():
    a = evaluate_image([g("Azure", (0, 0, 50, 20))], [p("Azure", (0, 2, 50, 18))], CFG)
    b = evaluate_image([g("DNS", (0, 0, 30, 20))], [], CFG)
    r = aggregate([a, b], CFG)
    assert r["categories"]["all"]["words"] == 2 and r["categories"]["all"]["detection_recall"] == 0.5


def test_consensus_status_is_scored_separately():
    gt = [g("AI", (0, 0, 20, 20)), g("Azure", (30, 0, 80, 20)), g("DNS", (90, 0, 120, 20))]
    pred = [PredWord("Al", (0, 2, 20, 18), 1.0, status="accepted"),       # wrong but accepted
            PredWord("Azure", (30, 2, 80, 18), 1.0, status="accepted"),
            PredWord("DN5", (90, 2, 120, 18), 0.4, status="uncertain")]  # goes to review
    r = aggregate([evaluate_image(gt, pred, CFG)], CFG)
    row = r["consensus_accepted"]
    assert row["accepted_words"] == 2 and row["accepted_cer"] == pytest.approx(1 / 7)
    assert row["review_rate"] == pytest.approx(1 / 3)


def test_single_engine_has_no_consensus_row():
    r = aggregate([evaluate_image([g("A", (0, 0, 9, 9))], [p("A", (0, 0, 9, 9))], CFG)], CFG)
    assert r["consensus_accepted"] is None


def test_predictions_inside_ignore_regions_are_not_false_positives():
    gt = [g("Azure", (0, 0, 50, 20))]
    pred = [p("Azure", (0, 2, 50, 18)), p("Microsoft", (100, 10, 180, 30)), p("III", (300, 0, 330, 20))]
    res = evaluate_image(gt, pred, CFG, ignore_regions=[(90, 0, 200, 40)])  # an embedded logo
    assert res.false_positives == 1


def test_scored_form_unifies_typography_and_strips_edge_punctuation():
    from eval.metrics import scored_text
    assert scored_text("What’s", CFG) == scored_text("What's", CFG) == "What's"
    assert scored_text("“Which", CFG) == "Which" and scored_text("-Data", CFG) == "Data"
    assert scored_text("Resource:", CFG) == "Resource" and scored_text("(if needed)", CFG) == "if needed"
    # inside a word punctuation stays: these are what gets anonymised
    for t in ("10.0.0.1", "j.kowalski@example.com", "api.example.com", "On-premises", "10.0.0.0/16"):
        assert scored_text(t, CFG) == t


def test_single_characters_and_punctuation_are_not_scored():
    gt = [g("Azure", (0, 0, 50, 20)), g("1", (60, 0, 70, 20)), g("…", (80, 0, 95, 20)), g("–", (100, 0, 110, 20))]
    pred = [p("Azure.", (0, 2, 50, 18)), p("l", (60, 2, 70, 18)), p("<>", (200, 0, 230, 20)), p("|", (300, 0, 305, 20))]
    r = aggregate([evaluate_image(gt, pred, CFG)], CFG)
    a = r["categories"]["all"]
    assert (a["words"], a["cer"], a["detection_recall"]) == (1, 0.0, 1.0)  # only "Azure", read right
    assert r["false_positives"] == 1  # "<>" is still invented text; "|" is too short to score


# --- character alignment (cfg.matching == "align") ------------------------------------

def test_engine_joining_words_is_not_an_error():
    # "(name," and "s" in the ground truth, one reading "(name, s" (c0001 on the test set).
    gt = [g("(name,", (0, 0, 60, 20)), g("s", (66, 0, 74, 20))]
    r = aggregate([evaluate_image(gt, [p("(name, s", (2, 2, 75, 18))], CFG)], CFG)
    assert r["categories"]["all"]["words"] == 1 and r["categories"]["all"]["wer"] == 0.0


def test_engine_joining_and_splitting_across_words():
    gt = [g("zone", (0, 0, 40, 20)), g("nonroutable", (46, 0, 140, 20)), g("spoke", (146, 0, 190, 20))]
    res = evaluate_image(gt, [p("zoneonroutablespoke", (0, 2, 190, 18))], CFG)
    by = {w["text"]: w for w in res.words}
    assert by["zone"]["exact"] and by["spoke"]["exact"]
    assert not by["nonroutable"]["exact"] and by["nonroutable"]["edits"] == 1


def test_shifted_box_with_the_right_text_is_correct():
    # The engine's box cuts the first letter and sticks out on the right: overlap well below 0.5 IoU.
    res = evaluate_image([g("WAN", (100, 0, 140, 20))], [p("WAN", (112, 3, 150, 19))], CFG)
    assert res.counts["all"].exact == 1 and res.false_positives == 0


def test_misread_missed_and_invented():
    gt = [g("AI", (0, 0, 20, 20)), g("Azure", (40, 0, 90, 20))]
    res = evaluate_image(gt, [p("Al", (0, 2, 20, 18)), p("III", (300, 0, 330, 20))], CFG)
    by = {w["text"]: w for w in res.words}
    assert by["AI"]["edits"] == 1 and by["AI"]["detected"]
    assert by["Azure"]["outcome"] == "missed" and by["Azure"]["edits"] == 5
    assert res.false_positives == 1
    assert {d["kind"] for d in res.disagreements} == {"mismatch", "missed", "extra"}


def test_reading_over_ignored_ground_truth_is_not_invented():
    res = evaluate_image([g("Poc", (0, 0, 30, 20), ignore=True)], [p("Poc", (0, 2, 30, 18))], CFG)
    assert res.false_positives == 0 and res.counts["all"].words == 0


def test_word_is_accepted_only_if_every_reading_behind_it_is():
    gt = [g("10.0.0.1", (0, 0, 80, 20))]
    pred = [PredWord("10.0.0.", (0, 2, 68, 18), 0.99, "accepted"), PredWord("1", (70, 2, 80, 18), 0.4, "uncertain")]
    [w] = evaluate_image(gt, pred, CFG).words
    assert w["exact"] and w["accepted"] is False and w["outcome"] == "review"


def test_vertical_text_in_either_direction():
    # Rotated label: words stacked vertically, one tall reading box.
    gt = [g("Azure", (100, 120, 120, 170)), g("Site", (100, 70, 120, 110)), g("Recovery", (100, 0, 120, 60))]
    for text in ("AzureSiteRecovery", "Azure Site Recovery"):
        res = evaluate_image(gt, [p(text, (100, 0, 120, 170))], CFG)
        assert res.counts["all"].exact == 3, text


def test_box_coverage_of_the_words_ink():
    import numpy as np
    img = np.full((40, 200, 3), 255, np.uint8)
    img[10:30, 20:80] = 0  # the word's ink
    gt = [g("Azure", (15, 5, 85, 35))]
    full = evaluate_image(gt, [p("Azure", (18, 8, 82, 32))], CFG, image=img)
    cut = evaluate_image(gt, [p("Azure", (32, 8, 82, 32))], CFG, image=img)  # first fifth cut off
    r_full, r_cut = (aggregate([x], CFG)["categories"]["all"] for x in (full, cut))
    assert r_full["box_full_coverage"] == 1.0 and r_full["box_mean_coverage"] == 1.0
    assert r_cut["box_full_coverage"] == 0.0 and abs(r_cut["box_mean_coverage"] - 0.8) < 0.01
    assert r_cut["wer"] == 0.0  # the reading is right; only the box is short


def test_word_matching_mode_is_still_available():
    import dataclasses
    word_cfg = dataclasses.replace(CFG, matching="word")
    r = aggregate([evaluate_image([g("(name,", (0, 0, 60, 20)), g("name2", (66, 0, 120, 20))],
                                  [p("(name, name2", (2, 2, 120, 18))], word_cfg)], word_cfg)
    assert r["categories"]["all"]["wer"] > 0  # one-to-one matching cannot pair a joined reading


def test_label_on_several_lines_in_any_reading_order():
    # "Top 16 techniques" stacked on three lines, read as one line by the engine.
    gt = [g("Top", (40, 0, 70, 20)), g("16", (45, 22, 65, 42)), g("techniques", (10, 44, 100, 64))]
    res = evaluate_image(gt, [p("Top 16 techniques", (10, 0, 100, 64))], CFG)
    assert res.counts["all"].exact == 3 and res.disagreements == []


def test_engine_word_boxes_in_reversed_order_along_a_rotated_line():
    # Rotated line read bottom-up: "Observer" below "monitoring"; the engine reads both words
    # right but puts their boxes the other way round.
    gt = [g("Observer", (100, 120, 120, 200)), g("monitoring", (100, 0, 120, 110))]
    pred = [p("Observer", (100, 0, 120, 80)), p("monitoring", (100, 90, 120, 200))]
    res = evaluate_image(gt, pred, CFG)
    assert res.counts["all"].exact == 2 and res.disagreements == []


def test_rows_of_a_table_are_not_one_group():
    from eval.metrics import _groups
    gt = [g("alpha", (0, 0, 100, 20)), g("beta", (0, 24, 100, 44)), g("gamma", (0, 48, 100, 68))]
    assert len(_groups(gt, [], CFG)) == 3
    # ... while the words of one row are
    row = [g("Azure", (0, 0, 50, 20)), g("Firewall", (56, 0, 130, 20))]
    assert len(_groups(row, [], CFG)) == 1


def test_stray_single_characters_of_the_engine_are_not_errors():
    gt = [g("Private", (0, 0, 60, 20)), g("Link", (66, 0, 100, 20)), g("service", (106, 0, 170, 20))]
    for pred in ([p("Private Link s service", (0, 2, 170, 18))],
                 [p("Private", (0, 2, 60, 18)), p("Link", (66, 2, 100, 18)), p("s", (101, 2, 105, 18)),
                  p("service", (106, 2, 170, 18))]):
        res = evaluate_image(gt, pred, CFG)
        assert res.counts["all"].exact == 3, pred
