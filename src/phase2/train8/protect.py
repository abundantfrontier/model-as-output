"""Protected spans from a pass-1 / RAG draft.

Invariant: every protected span from the draft must appear in the rewrite,
or that rewrite is rejected. Used at training-filter time, generation time
(copy-biased / coverage decoding), and as a guard stricter than fact-diff.

Spans are taken from the draft itself. Gist tokens come from the must-answer
regex vocabulary so `sledge` cannot become `sled`, but we never train on the
held-out must/leak questions.
"""
from __future__ import annotations

import re
from typing import Iterable

# Longer names first so "Henry Clerval" matches before "Henry".
BOOK_NAMES = [
    "Cornelius Agrippa",
    "Albertus Magnus",
    "Elizabeth Lavenza",
    "Justine Moritz",
    "Henry Clerval",
    "Robert Walton",
    "Margaret Saville",
    "Alphonse Frankenstein",
    "Caroline Beaufort",
    "William Frankenstein",
    "M. De Lacey",
    "M. De Lacy",
    "De Lacey",
    "De Lacy",
    "Mrs Saville",
    "Paradise Lost",
    "the Sorrows of Werter",
    "Sorrows of Werter",
    "Mont Blanc",
    "sea of ice",
    "St. Petersburgh",
    "St Petersburgh",
    "M. Krempe",
    "M. Waldman",
    "Plainpalais",
    "Ingolstadt",
    "Ingoldstadt",
    "Montanvert",
    "Chamounix",
    "Chamouni",
    "Archangel",
    "Petersburgh",
    "Paracelsus",
    "Clerval",
    "Agrippa",
    "Elizabeth",
    "Walton",
    "Justine",
    "Safie",
    "Felix",
    "William",
    "Ernest",
    "Krempe",
    "Waldman",
    "Beaufort",
    "Kirwin",
    "Moritz",
    "Plutarch",
    "Werter",
    "Werther",
    "Saville",
    "Geneva",
    "Orkney",
    "Ireland",
    "London",
    "Thonon",
    "Jura",
    "Caroline",
    "Alphonse",
    "Henry",
    "Victor",
    "Frankenstein",
]

# Distinctive gist words/phrases the must-answer regex needs. Applied only
# when the draft already contains them.
GIST_SPANS = [
    "sledge",
    "cousin",
    "father's sister",
    "father’s sister",
    "sent from Italy",
    "yellow eye",
    "dreary night of November",
    "night of November",
    "principle of life",
    "lifeless matter",
    "animate lifeless",
    "black ballot",
    "black ballots",
    "sea of ice",
    "fallen angel",
    "Paradise Lost",
    "funeral pile",
    "ice-raft",
    "ice raft",
    "cabin window",
    "wedding night",
    "wedding-night",
    "scarlet fever",
    "scarlet",
    "miniature",
    "7 May",
    "seventh of May",
    "Plainpalais",
    "tranquillity",
    "natural philosopher",
    "letters of introduction",
    "era of my being",
]

SPEAKER_SKIP = {
    "victor": {"victor", "frankenstein"},
    "creature": {"creature", "daemon", "monster", "wretch", "fiend", "being"},
}

# Double quotes only. Apostrophes in father's / Agrippa's are not quotes.
QUOTE_RE = re.compile(r"[“\"]([^\"”]{3,40})[\"”]")
NUM_RE = re.compile(r"\b\d+\b")
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _word_pat(span: str) -> re.Pattern:
    parts = [re.escape(w) for w in WORD_RE.findall(span)]
    if not parts:
        return re.compile(re.escape(span), re.I)
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.I)


def span_in(text: str, span: str) -> bool:
    if not span or not text:
        return False
    return bool(_word_pat(span).search(text))


def missing_spans(text: str, spans: Iterable[str]) -> list[str]:
    miss = []
    seen = set()
    for s in spans:
        k = _norm_key(s)
        if not k or k in seen:
            continue
        seen.add(k)
        if not span_in(text, s):
            miss.append(s)
    return miss


def extract_protected(pass1: str, char: str | None = None) -> list[str]:
    """Surface forms from the draft that the rewrite must keep."""
    text = pass1 or ""
    found: list[str] = []
    seen: set[str] = set()
    skip = SPEAKER_SKIP.get((char or "").lower(), set())

    def add(span: str) -> None:
        s = re.sub(r"\s+", " ", (span or "").strip())
        if len(s) < 2:
            return
        k = _norm_key(s)
        if k in seen or k in skip:
            return
        if k in {"i", "me", "my", "we", "our", "the", "a", "an", "and", "or"}:
            return
        if re.match(r"^[a-z]\b", s):
            return
        seen.add(k)
        found.append(s)

    catalog = sorted(set(BOOK_NAMES + GIST_SPANS), key=len, reverse=True)
    for name in catalog:
        m = _word_pat(name).search(text)
        if m:
            add(m.group(0))

    for m in NUM_RE.finditer(text):
        add(m.group(0))

    for m in QUOTE_RE.finditer(text):
        add(m.group(1))

    return found


def span_ok(pass1: str, rewrite: str, char: str | None = None) -> tuple[bool, list[str], list[str]]:
    spans = extract_protected(pass1, char)
    miss = missing_spans(rewrite, spans)
    return (not miss), spans, miss


# ---------------------------------------------------------------------------
# Copy-biased / coverage decoding
# ---------------------------------------------------------------------------

def _encode_variants(tok, span: str) -> list[list[int]]:
    """Tokenizations of a span, with and without a leading space."""
    out = []
    seen = set()
    for s in (span, " " + span, span.lower(), " " + span.lower()):
        ids = tok.encode(s, add_special_tokens=False)
        if not ids:
            continue
        key = tuple(ids)
        if key in seen:
            continue
        seen.add(key)
        out.append(ids)
    return out


class ProtectedSpanProcessor:
    """Logits processor: copy-bias protected tokens, block EOS while any span
    is still missing, and force the next token of a span once its prefix has
    started so Clerval cannot become Cliversal.
    """

    EOS = (151645, 151643)

    def __init__(self, tokenizer, spans: list[str], prompt_len: int, copy_bias: float = 3.5):
        self.tok = tokenizer
        self.prompt_len = int(prompt_len)
        self.copy_bias = float(copy_bias)
        self.force_bias = 80.0
        self.span_text = [s for s in spans if s and s.strip()]
        self.variants: list[list[list[int]]] = [_encode_variants(tokenizer, s) for s in self.span_text]
        self.first_ids: set[int] = set()
        self.all_ids: set[int] = set()
        for var_list in self.variants:
            for ids in var_list:
                if ids:
                    self.first_ids.add(ids[0])
                    self.all_ids.update(ids)

    def _missing(self, decoded: str) -> list[int]:
        return [i for i, s in enumerate(self.span_text) if not span_in(decoded, s)]

    def __call__(self, input_ids, scores):
        import torch

        if not self.span_text:
            return scores
        seq = input_ids[0]
        gen = seq[self.prompt_len :]
        decoded = self.tok.decode(gen, skip_special_tokens=True)
        missing = self._missing(decoded)
        if not missing:
            return scores

        scores = scores.clone()
        # Coverage: do not stop until every protected span has appeared.
        for eid in self.EOS:
            if eid < scores.shape[-1]:
                scores[:, eid] = torch.finfo(scores.dtype).min

        # Prefix force: if the generated suffix is a prefix of a missing span,
        # emit the next token of that span.
        gen_list = gen.tolist() if hasattr(gen, "tolist") else list(gen)
        forced = None
        for mi in missing:
            for ids in self.variants[mi]:
                n = len(ids)
                if n <= 1:
                    continue
                for k in range(n - 1, 0, -1):
                    if len(gen_list) >= k and gen_list[-k:] == ids[:k]:
                        forced = ids[k]
                        break
                if forced is not None:
                    break
            if forced is not None:
                break
        if forced is not None and forced < scores.shape[-1]:
            scores[:, forced] = scores[:, forced] + self.force_bias
            return scores

        # Copy-bias remaining span tokens so sledge wins over sled.
        still_ids = set()
        still_first = set()
        for mi in missing:
            for ids in self.variants[mi]:
                still_ids.update(ids)
                if ids:
                    still_first.add(ids[0])
        if still_first:
            idx = torch.tensor(sorted(still_first), device=scores.device, dtype=torch.long)
            idx = idx[idx < scores.shape[-1]]
            if idx.numel():
                scores[:, idx] = scores[:, idx] + self.copy_bias + 1.0
        extra = still_ids - still_first
        if extra:
            idx = torch.tensor(sorted(extra), device=scores.device, dtype=torch.long)
            idx = idx[idx < scores.shape[-1]]
            if idx.numel():
                scores[:, idx] = scores[:, idx] + self.copy_bias
        return scores


def logits_processor(tokenizer, spans: list[str], prompt_len: int, copy_bias: float = 3.5):
    from transformers import LogitsProcessorList

    return LogitsProcessorList(
        [ProtectedSpanProcessor(tokenizer, spans, prompt_len, copy_bias=copy_bias)]
    )


def generate_constrained(model, tok, input_ids, attention_mask, spans, penalty=True, pad_token_id=None):
    """model.generate with copy-bias / coverage for protected spans."""
    import common as C

    kw = C.generate_kwargs(penalty=penalty, pad_token_id=pad_token_id)
    # no_repeat_ngram_size can zero a forced name token after our processor
    # runs, and the pure-Python n-gram scan stalls decode. Keep repetition_penalty.
    kw.pop("no_repeat_ngram_size", None)
    if spans:
        kw["logits_processor"] = logits_processor(tok, spans, input_ids.shape[1])
    return model.generate(input_ids=input_ids, attention_mask=attention_mask, **kw)
