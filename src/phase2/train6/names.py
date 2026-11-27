#!/usr/bin/env python3
"""Canonical proper nouns from the 1818 text, and a near-name rejector.

Starts from protect.BOOK_NAMES. Adds other proper nouns that actually appear
in the chapter text (and the source dump, if present). Spelling variants
printed in 1818 (Chamounix/Chamouni, De Lacey/De Lacy, Ingolstadt/Ingoldstadt,
Werter) stay canonical because they are in the book or in BOOK_NAMES.

A near-name is a token that is not in the book vocabulary and not canonical,
but is a close misspelling of a canonical name. Known drifts are always
rejected: Clervaux, Cliversal, Morritz, Cornelis, Agrippus, Montanvirt.
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
from protect import BOOK_NAMES  # noqa: E402

OUT = C.DATA / "canonical_names.txt"

# Always a miss, even if edit distance would miss it (Cliversal vs Clerval).
BAD_NEAR = {
    "clervaux",
    "cliversal",
    "clerival",
    "cleraval",
    "clervals",
    "morritz",
    "moritzz",
    "cornelis",
    "agrippus",
    "agrippas",
    "montanvirt",
    "montanverde",
    "frankstein",
    "frankenstien",
    "frankenstine",
    "frankestien",
    "lavenzaa",
    "ingolstat",
    "waldmann",
    "krempee",
    "beauford",
}

# Capitalized common nouns / function words. Not proper names.
STOP = {
    "a", "an", "the", "and", "or", "but", "if", "when", "while", "after", "before",
    "then", "this", "that", "these", "those", "his", "her", "he", "she", "they",
    "we", "you", "my", "our", "their", "its", "it", "in", "on", "at", "to", "for",
    "of", "with", "from", "by", "as", "not", "no", "yes", "there", "here", "so",
    "yet", "nor", "what", "which", "who", "whom", "whose", "where", "how", "why",
    "all", "some", "any", "one", "two", "three", "i", "oh", "ah", "mr", "mrs",
    "st", "dr", "volume", "chapter", "letter", "preface", "sir", "madam",
    "god", "heaven", "death", "nature", "science", "life", "time", "man", "men",
    "love", "hope", "fear", "day", "night", "father", "mother", "son", "daughter",
    "brother", "sister", "friend", "child", "children", "being", "creature",
    "monster", "fiend", "daemon", "wretch", "stranger", "captain", "professor",
    "english", "latin", "french", "german", "italian", "arabian", "swiss",
    "every", "during", "however", "perhaps", "although", "therefore", "thus",
    "upon", "into", "over", "under", "about", "among", "between", "without",
    "within", "again", "still", "only", "even", "also", "such", "same", "other",
    "great", "little", "first", "last", "next", "own", "many", "much", "more",
    "most", "very", "could", "would", "should", "shall", "will", "may", "might",
    "must", "have", "has", "had", "was", "were", "been", "being", "are", "is",
    "am", "do", "did", "does", "their", "them", "him", "her", "me", "us",
    "my", "mine", "your", "yours", "our", "ours",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "gutenberg", "prometheus", "project", "political", "justice",
    "come", "dear", "doth", "good", "let", "sal", "sea", "lost", "lives",
    "north", "mont", "turk", "greek", "arabic", "turkish", "scotch",
    "christian", "creator", "madame", "englishman", "haunted", "unborrowed",
    "miss", "mr", "mrs",
}
# Titles that may introduce a real name but are not names by themselves.
TITLE = {"de", "madame", "miss", "mrs", "mr", "st", "m"}

WORD = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’\-]*")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _sources() -> list[Path]:
    """Chapter files are the 1818 text the reader sees.

    source/*.txt is the same novel plus a Project Gutenberg wrapper, which
    would add 'PROJECT GUTENBERG' and similar non-names. Chapters are the scan.
    """
    return sorted((C.ROOT / "chapters").glob("*.md"))


def book_text() -> str:
    parts = []
    for p in _sources():
        parts.append(p.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


def _clean_token(tok: str) -> str:
    t = tok.replace("’", "'").strip("'\"“”").lower()
    if t.endswith("'s"):
        t = t[:-2]
    return t


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 99
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


@lru_cache(maxsize=1)
def book_vocab() -> set[str]:
    vocab = set()
    for m in WORD.finditer(book_text()):
        t = _clean_token(m.group(0))
        if t:
            vocab.add(t)
    return vocab


def discover_names(text: str) -> list[str]:
    """Mid-sentence proper nouns that recur, plus every BOOK_NAMES entry.

    Bigrams are kept only when both parts recur (Henry Clerval) or the pair
    is already in BOOK_NAMES. One-off capitals ('Haunted', 'Unborrowed') stay out.
    """
    counts: dict[str, int] = {}
    surface: dict[str, str] = {}
    bigrams: dict[tuple[str, str], int] = {}
    for sent in SENT_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        sent = re.sub(r"^#+\s*", "", sent)
        toks = WORD.findall(sent)
        if not toks:
            continue
        cased = []
        for i, tok in enumerate(toks):
            raw = tok.strip("'’")
            if len(raw) < 4 or i == 0 or not raw[0].isupper() or raw.isupper():
                cased.append(None)
                continue
            if raw.endswith("'s") or raw.endswith("’s"):
                raw = raw[:-2]
            key = _clean_token(raw)
            if not key or key in STOP or key in TITLE or len(key) < 4:
                cased.append(None)
                continue
            counts[key] = counts.get(key, 0) + 1
            surface.setdefault(key, raw)
            cased.append(key)
        for a, b in zip(cased, cased[1:]):
            if a and b:
                bigrams[(a, b)] = bigrams.get((a, b), 0) + 1

    found = set(BOOK_NAMES)
    for key, n in counts.items():
        if n >= 2 and key not in STOP:
            found.add(surface[key])
    for (a, b), n in bigrams.items():
        if n < 1:
            continue
        if a in TITLE or b in TITLE or a in STOP or b in STOP:
            continue
        # A repeated pair, or a one-mention full name (Louis Manoir).
        if n >= 2 or (counts.get(a, 0) >= 1 and counts.get(b, 0) >= 1 and len(a) >= 4 and len(b) >= 4):
            if a not in STOP:
                found.add(surface[a])
            if b not in STOP:
                found.add(surface[b])
            if a not in STOP and b not in STOP:
                found.add(f"{surface[a]} {surface[b]}")
    return sorted(found, key=lambda s: (-len(s), s.lower()))


def load_canonical() -> list[str]:
    if OUT.is_file():
        rows = []
        for line in OUT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line)
        if rows:
            return rows
    return list(BOOK_NAMES)


@lru_cache(maxsize=1)
def canonical_tokens() -> set[str]:
    toks = set()
    for name in load_canonical():
        for part in WORD.findall(name):
            t = _clean_token(part)
            if len(t) >= 4:
                toks.add(t)
    return toks


def names_in(text: str) -> list[str]:
    """Canonical names that occur in text, longest first, each at most once."""
    found = []
    seen = set()
    for name in load_canonical():
        k = name.lower()
        if k in seen:
            continue
        # Local import keeps span rules identical to the teacher guard.
        from protect import span_in

        if span_in(text or "", name):
            seen.add(k)
            found.append(name)
    return found


def near_name_tokens(text: str) -> list[str]:
    """Non-canonical close misspellings. Empty means the text is name-clean."""
    if not text:
        return []
    vocab = book_vocab()
    ctoks = canonical_tokens()
    # Fuzzy targets: canonical tokens long enough to be distinctive.
    targets = [t for t in ctoks if len(t) >= 5]
    bad = []
    seen = set()
    for m in WORD.finditer(text):
        raw = m.group(0)
        t = _clean_token(raw)
        if not t or t in seen:
            continue
        seen.add(t)
        if t in BAD_NEAR:
            bad.append(raw)
            continue
        if t in vocab or t in ctoks or t in STOP:
            continue
        stems = [t]
        # "Franksteins" / "Clervals": a trailing s on a near-name or a real name.
        if len(t) >= 6 and t.endswith("s"):
            stems.append(t[:-1])
        hit = False
        for stem in stems:
            if stem in BAD_NEAR or (stem != t and stem in ctoks):
                hit = True
                break
            if stem in vocab or stem in STOP or len(stem) < 5:
                continue
            best = 99
            for c in targets:
                if c[0] != stem[0] or abs(len(c) - len(stem)) > 2:
                    continue
                d = levenshtein(stem, c)
                if d < best:
                    best = d
                    if best <= 1:
                        break
            if best == 1 or (best == 2 and len(stem) >= 8):
                hit = True
                break
        if hit:
            bad.append(raw)
    return bad


def missing_canonical(source: str, target: str) -> list[str]:
    """Canonical names present in source and absent from target."""
    from protect import span_in

    miss = []
    for name in names_in(source):
        if not span_in(target or "", name):
            miss.append(name)
    return miss


def write_list() -> dict:
    text = book_text()
    names = discover_names(text)
    # Drop stopword-only singles that slipped in.
    cleaned = []
    for name in names:
        parts = [_clean_token(p) for p in WORD.findall(name)]
        parts = [p for p in parts if p]
        if not parts:
            continue
        if all(p in STOP for p in parts):
            continue
        cleaned.append(name)
    # Stable unique, longest first so readers see phrases before pieces.
    uniq = []
    seen = set()
    for name in sorted(set(cleaned), key=lambda s: (-len(s), s.lower())):
        k = name.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(name)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(uniq) + "\n", encoding="utf-8")
    base = {n.lower() for n in BOOK_NAMES}
    added = [n for n in uniq if n.lower() not in base]
    return {"n": len(uniq), "added": added, "path": str(OUT)}


def _selftest() -> None:
    write_list()
    # Refresh caches after the file exists.
    book_vocab.cache_clear()
    canonical_tokens.cache_clear()
    samples = {
        "Clervaux": True,
        "I travelled with Clervaux.": True,
        "Cliversal was my friend.": True,
        "Justine Morritz stood there.": True,
        "I read Cornelis Agrippa.": True,
        "Agrippus filled the page.": True,
        "We climbed Montanvirt.": True,
        "My name is Victor Frankstein.": True,
        "I am Victor Franksteins.": True,
        "Henry Clerval and Justine Moritz met at Montanvert.": False,
        "Cornelius Agrippa is sad trash.": False,
        "I am by birth a Genevese.": False,
        "Elizabeth Lavenza was my cousin.": False,
        "The sorrows of Werter lay on the table.": False,
        "Ingolstadt and Ingoldstadt are both printed.": False,
        "Chamounix and Chamouni are the same valley.": False,
    }
    failed = 0
    for text, expect in samples.items():
        got = bool(near_name_tokens(text))
        if got != expect:
            failed += 1
            print("FAIL", expect, got, near_name_tokens(text), text)
        else:
            print("ok", "near" if got else "clean", text[:60])
    if failed:
        raise SystemExit(f"{failed} near-name selftests failed")
    print("selftest ok", len(load_canonical()), "names")


if __name__ == "__main__":
    info = write_list()
    print("wrote", info["path"], "n", info["n"], "added", len(info["added"]))
    print("ADDED:")
    for n in info["added"]:
        print(" ", n)
    _selftest()
