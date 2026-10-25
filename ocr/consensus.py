"""Two engines and a consensus (CLAUDE.md, decision 3).

The primary engine finds and reads the words. The secondary engine reads the same boxes, so
both votes concern exactly the same pixels. Then, per word:

- both read the same text, both confident enough           -> accepted ("agree");
- otherwise both read it again from an enlarged crop; if at least ``reread_min_votes`` of the
  four readings are identical                                -> accepted ("reread_agree");
  but a majority that overrules the primary reading with a text of another length
  (more than ``reread_max_length_change`` characters)        -> uncertain ("reread_length"),
  or with letters dropped from its edges ("Linux" -> "inux")  -> uncertain ("reread_edge_letter");
- otherwise                                                  -> uncertain, for human review
  ("engines_disagree", or "low_confidence" when they agreed but were unsure);
- accepted words matching a flag rule (look-alike glyphs, dashes at the edges) become
  uncertain too ("ambiguous_glyphs", "edge_dash"): both engines make those errors alike;
  so does "MDTO" when the image also has "MDT1" (apply_sibling_flags).

Uncertain words keep the primary engine's reading. Every reading is kept in ``votes``.
Optionally, text found only by the secondary engine is added as uncertain
("secondary_only"), trading review work for detection completeness.

A consensus cannot catch an error both engines make the same way (e.g. "AI" read as "Al" in
fonts where I and l look alike); the evaluation measures how often that happens.
"""
from __future__ import annotations

import re
from collections import Counter

from PIL import Image

from ocr.config import Config, ConsensusConfig, load_config
from ocr.engines.base import Box, OcrEngine, OcrWord
from ocr.textnorm import TYPOGRAPHY, alnum_count, has_latin_letter


def _overlap(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / small if small > 0 else 0.0


# Look-alike glyphs that both engines get wrong the same way, in UI sans fonts:
# - "l" where an uppercase "I" fits the casing: "Al", "OpenAl", "loT" (observed errors are all
#   I read as l); "I" squeezed between lowercase letters for the opposite case;
# - O/0 and l/1 against the other class inside tokens mixing letters and digits ("MDT0").
_I_AS_L = re.compile(r"[A-Z]l(?![a-z])|^l[a-z]*[A-Z]|[a-z]I[a-z]"
                     # "Id" at the end of a camelCase name read as "ld" after a consonant (tenantld,
                     # Appld; not build, field, world)
                     r"|[bcdfghjkmnpqstvwxz]ld(?=\W*$)")
_MIXED_CONFUSABLE = re.compile(r"[A-Za-z]0|0[A-Za-z]|\dO|O\d|\dl|l\d")


def _ambiguous(token: str) -> bool:
    if _I_AS_L.search(token):
        return True
    mixed = any(c.isalpha() for c in token) and any(c.isdigit() for c in token)
    return mixed and bool(_MIXED_CONFUSABLE.search(token))


_EDGE_DASH = re.compile(r"^[-–—_|]|[-–—|]$")


def secondary_as(secondary: str, primary: str, cfg: ConsensusConfig) -> str:
    """The secondary reading as it counts against the primary one.

    Tesseract adds characters at the edges of a word crop where frames, lines and circles touch
    the text ("Azure |", "Private.", and with secondary_edge_brackets "(12)" for a number in a
    circle). If the secondary reading
    equals the primary one once such characters are stripped from its edges, it counts as the
    primary reading. Each edge on its own: where the primary reading has such a character at
    that edge ("“Deploy"), the edge must match exactly. With normalize_typography, typographic
    variants count as equal (" = “ = ”, ' = ’, - = –). With secondary_edge_tokens, a single
    character read apart (after a space) at either edge does not count either.
    """
    if secondary == primary:
        return secondary
    norm = (lambda t: t.translate(TYPOGRAPHY)) if cfg.normalize_typography else (lambda t: t)
    s, p = norm(secondary), norm(primary)
    if s == p:
        return primary
    junk = cfg.secondary_edge_junk

    def stripped(t: str) -> str:
        if p:
            if p[0] not in junk:
                t = t.lstrip(" " + junk)
                if cfg.secondary_edge_brackets and p[0] != "(" and t.startswith("("):
                    t = t[1:].lstrip(" " + junk)
            if p[-1] not in junk:
                t = t.rstrip(" " + junk)
                if cfg.secondary_edge_brackets and p[-1] != ")" and t.endswith(")"):
                    t = t[:-1].rstrip(" " + junk)
        return t.strip(" ")

    if stripped(s) == p:
        return primary
    if cfg.secondary_edge_tokens and " " not in p:
        # A one-character token the secondary engine read apart at the crop's edge is a sliver
        # of the neighbouring word or of a line ("Understand 1", "| agents i").
        tokens = s.split()
        if len(tokens) > 1 and len(tokens[0]) == 1:
            tokens = tokens[1:]
        if len(tokens) > 1 and len(tokens[-1]) == 1:
            tokens = tokens[:-1]
        if len(tokens) == 1 and stripped(tokens[0]) == p:
            return primary
    return secondary


def revote(word: OcrWord, cfg: ConsensusConfig) -> OcrWord:
    """The decision made again from a word's saved votes (primary, secondary, and when there was
    a re-read, both enlarged readings, in that order), with the current rules. The secondary
    engine's confidence is not saved and counts as sufficient. For rescoring saved runs."""
    v = list((word.votes or {}).items())
    if len(v) < 2 or word.reason in ("secondary_only", "native"):
        return word
    p, s = v[0][1], v[1][1]
    keep = dict(word.votes)
    if p == secondary_as(s, p, cfg) and word.conf >= cfg.primary_min_conf:
        return OcrWord(p, word.bbox_px, word.conf, word.char_conf if len(word.char_conf) == len(p) else [word.conf] * len(p),
                       "accepted", "agree", keep)
    if len(v) < 4:
        return word
    p2, s2 = v[2][1], v[3][1]
    agree = p == secondary_as(s, p, cfg)
    counted = [p, secondary_as(s, p, cfg), p2, secondary_as(s2, p, cfg)]
    text, n = Counter(t for t in counted if t).most_common(1)[0] if any(counted) else ("", 0)
    cc = [word.conf] * len(p)
    if text and n >= cfg.reread_min_votes:
        if overrule_too_long(p, text, cfg):
            return OcrWord(p, word.bbox_px, word.conf, cc, "uncertain", "reread_length", keep)
        if drops_edge_letters(p, text, cfg):
            return OcrWord(p, word.bbox_px, word.conf, cc, "uncertain", "reread_edge_letter", keep)
        return OcrWord(text, word.bbox_px, word.conf, [word.conf] * len(text), "accepted", "reread_agree", keep)
    return OcrWord(p, word.bbox_px, word.conf, cc, "uncertain", "low_confidence" if agree else "engines_disagree", keep)


def settle_short(word: OcrWord, cfg: ConsensusConfig) -> OcrWord:
    """A word with fewer than review_min_alnum letters and digits and no Latin letter ("1", "→",
    an icon read as "个") is never anonymised: it is not sent to review - unless some reading saw
    more there (a fragment like '"i of "EncryptResponse" is a misread long word, not a short
    one). A lone Latin letter goes to review: it may be what is left of a word ("D" for "ID")."""
    if word.status == "accepted" or alnum_count(word.text) >= cfg.review_min_alnum or has_latin_letter(word.text):
        return word
    if any(alnum_count(v) >= cfg.review_min_alnum for v in (word.votes or {}).values()):
        return word
    return OcrWord(word.text, word.bbox_px, word.conf, word.char_conf, "accepted", "short", word.votes)


def flag_reason(text: str, cfg: ConsensusConfig) -> str | None:
    """Why an agreed reading still needs a human, or None."""
    t = text.strip(",;:.!?()[]{}\"'")
    if cfg.flag_ambiguous_glyphs and _ambiguous(t):
        return "ambiguous_glyphs"
    if cfg.flag_edge_dashes and _EDGE_DASH.search(text) and len(text) > 1:
        return "edge_dash"
    return None


def apply_flags(word: OcrWord, cfg: ConsensusConfig) -> OcrWord:
    """Turn an accepted word into an uncertain one if a flag rule hits."""
    if word.status != "accepted":
        return word
    reason = flag_reason(word.text, cfg)
    if reason is None:
        return word
    return OcrWord(word.text, word.bbox_px, word.conf, word.char_conf, "uncertain", reason, word.votes)


def overrule_too_long(primary: str, majority: str, cfg: ConsensusConfig) -> bool:
    """The re-read majority replaces the primary reading with one of a different length."""
    return (cfg.reread_max_length_change >= 0 and majority != primary
            and abs(len(majority) - len(primary)) > cfg.reread_max_length_change)


def drops_edge_letters(primary: str, majority: str, cfg: ConsensusConfig) -> bool:
    """The majority is the primary reading without one or two letters or digits at its edges."""
    if not cfg.reread_keep_edge_letters or len(majority) >= len(primary) or not majority:
        return False
    n = len(primary)
    for a in range(3):
        for b in range(3 - a):
            if (a or b) and primary[a:n - b] == majority:
                return any(ch.isalnum() for ch in primary[:a] + primary[n - b:])
    return False


_LETTER_O = re.compile(r"^([A-Z]{1,6})O$")


def apply_sibling_flags(words: list[OcrWord], cfg: ConsensusConfig) -> list[OcrWord]:
    """O and 0 look alike: an accepted "MDTO" or "ATO" next to "MDT1" or "AT1" in the same image
    is most likely "MDT0" / "AT0" and goes to review."""
    if not cfg.flag_ambiguous_glyphs:
        return words
    texts = {w.text for w in words}
    out = []
    for w in words:
        m = _LETTER_O.match(w.text) if w.status == "accepted" else None
        if m and any(re.fullmatch(re.escape(m.group(1)) + r"\d+", t) for t in texts):
            w = OcrWord(w.text, w.bbox_px, w.conf, w.char_conf, "uncertain", "ambiguous_glyphs", w.votes)
        out.append(w)
    return out


def _joined(words: list[OcrWord]) -> tuple[str, float]:
    """One reading of a box: its words' text joined by spaces, and the lowest confidence."""
    if not words:
        return "", 0.0
    return " ".join(w.text for w in words), min(w.conf for w in words)


class ConsensusEngine(OcrEngine):
    name = "consensus"

    def __init__(self, cfg: ConsensusConfig | None = None, config: Config | None = None,
                 primary: OcrEngine | None = None, secondary: OcrEngine | None = None):
        from ocr.engines.registry import create

        config = config or load_config()
        self.cfg = cfg or config.consensus
        self.primary = primary or create(self.cfg.primary, config=config)
        self.secondary = secondary or create(self.cfg.secondary, config=config)

    def versions(self) -> dict[str, str]:
        v = {f"{self.primary.name}.{k}": str(val) for k, val in self.primary.versions().items()}
        v.update({f"{self.secondary.name}.{k}": str(val) for k, val in self.secondary.versions().items()})
        return v

    def detect(self, img: Image.Image) -> list[Box]:
        return self.primary.detect(img)

    def recognize(self, img: Image.Image, boxes: list[Box]) -> list[OcrWord]:
        return self._decide_all(img, self.primary.recognize(img, boxes))

    def read(self, img: Image.Image) -> list[OcrWord]:
        words = apply_sibling_flags(self._decide_all(img, self.primary.read(img)), self.cfg)
        if self.cfg.secondary_detection:
            words += self._secondary_only(img, words)
        return words

    # --- per word ---------------------------------------------------------------------

    def _reread(self, img: Image.Image, box: Box) -> tuple[str, str]:
        """Both engines read the word again from a crop enlarged ``reread_scale`` times."""
        pad, s = self.cfg.reread_pad_px, self.cfg.reread_scale
        x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
        x1, y1 = min(img.width, int(box[2]) + pad + 1), min(img.height, int(box[3]) + pad + 1)
        crop = img.crop((x0, y0, x1, y1))
        big = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)
        whole = (0.0, 0.0, float(big.width), float(big.height))
        p2, _ = _joined(self.primary.recognize(big, [whole]))
        s2, _ = _joined(self.secondary.recognize(big, [whole]))
        return p2, s2

    def _decide_all(self, img: Image.Image, primary_words: list[OcrWord]) -> list[OcrWord]:
        # The secondary engine reads all boxes in one call (in parallel where it can).
        second = self.secondary.recognize_each(img, [p.bbox_px for p in primary_words])
        return [settle_short(apply_flags(self._vote(img, p, _joined(s)), self.cfg), self.cfg)
                for p, s in zip(primary_words, second)]

    def _vote(self, img: Image.Image, p: OcrWord, secondary_reading: tuple[str, float]) -> OcrWord:
        c = self.cfg
        s_text, s_conf = secondary_reading
        votes = {self.primary.name: p.text, self.secondary.name: s_text}
        agree = p.text == secondary_as(s_text, p.text, c)
        if agree and p.conf >= c.primary_min_conf and s_conf >= c.secondary_min_conf:
            return OcrWord(p.text, p.bbox_px, min(p.conf, s_conf), p.char_conf, "accepted", "agree", votes)

        p2, s2 = self._reread(img, p.bbox_px)
        votes[f"{self.primary.name}@x{c.reread_scale:g}"] = p2
        votes[f"{self.secondary.name}@x{c.reread_scale:g}"] = s2
        # Votes keep the raw readings; counting uses the secondary's readings without edge junk.
        counted = [p.text, secondary_as(s_text, p.text, c), p2, secondary_as(s2, p.text, c)]
        text, n = Counter(t for t in counted if t).most_common(1)[0] if any(counted) else ("", 0)
        if text and n >= c.reread_min_votes:
            if overrule_too_long(p.text, text, c):
                return OcrWord(p.text, p.bbox_px, p.conf, p.char_conf, "uncertain", "reread_length", votes)
            if drops_edge_letters(p.text, text, c):
                return OcrWord(p.text, p.bbox_px, p.conf, p.char_conf, "uncertain", "reread_edge_letter", votes)
            char_conf = p.char_conf if text == p.text else [p.conf] * len(text)
            return OcrWord(text, p.bbox_px, p.conf, char_conf, "accepted", "reread_agree", votes)
        reason = "low_confidence" if agree else "engines_disagree"
        return OcrWord(p.text, p.bbox_px, p.conf, p.char_conf, "uncertain", reason, votes)

    def _secondary_only(self, img: Image.Image, primary_words: list[OcrWord]) -> list[OcrWord]:
        extra = []
        for w in self.secondary.read(img):
            if all(_overlap(w.bbox_px, p.bbox_px) <= self.cfg.secondary_max_overlap for p in primary_words):
                extra.append(OcrWord(w.text, w.bbox_px, w.conf, w.char_conf, "uncertain", "secondary_only",
                                     {self.secondary.name: w.text}))
        return extra
