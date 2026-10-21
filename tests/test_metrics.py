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
    assert res.counts["all"].edits == 1


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
