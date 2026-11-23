import dataclasses

import pytest

import ocr.anonymize as anon
from ocr.anonymize import anonymize, anonymize_sections, available
from ocr.config import load_config

CFG = load_config().anonymize

needs_model = pytest.mark.skipif(not available(CFG), reason="spaCy model not installed (./setup.sh --anonymize)")


@pytest.fixture
def regex_only(monkeypatch):
    """No spaCy: only the regular expressions run."""
    monkeypatch.setattr(anon, "_pipeline", lambda model: None)


def test_regular_expressions_replace_emails_phones_authors_and_listed_companies(regex_only):
    got = anonymize("Author: Jan Kowalski\nMail j.kowalski@example.com or call +48 601 234 567.\n"
                    "Avenga Capital and avenga.com work with Qinshift Group.", CFG)
    assert "Author: [FULL NAME]" in got and "[EMAIL]" in got and "[PHONE NUMBER]" in got
    assert got.count("[BIZ_ENTITY]") == 3 and "Avenga" not in got and "Qinshift" not in got
    assert "Kowalski" not in got and "example.com" not in got


def test_sections_keep_their_headers(regex_only):
    out = anonymize_sections("=== a.png ===\nmail: x@y.pl\n\n=== b.pdf | page 1 | p1r1 vector_diagram ===\nnothing\n",
                             CFG)
    assert out == "=== a.png ===\nmail: [EMAIL]\n\n=== b.pdf | page 1 | p1r1 vector_diagram ===\nnothing\n"


def test_company_list_comes_from_the_config(regex_only):
    cfg = dataclasses.replace(CFG, target_companies=("Contoso",))
    assert anonymize("Contoso and Avenga", cfg) == "[BIZ_ENTITY] and Avenga"


@needs_model
def test_names_organisations_places_and_roles_with_the_model():
    got = anonymize("Jan Kowalski, Head of Security at Microsoft, met Anna Nowak in Warsaw.", CFG)
    assert "Kowalski" not in got and "Nowak" not in got and "Warsaw" not in got and "Microsoft" not in got
    assert "[FULL NAME]" in got and "[LOCATION]" in got
