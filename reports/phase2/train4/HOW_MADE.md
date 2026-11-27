# How train4 single-pass SFT data was made

Teacher is the train3 guarded pipeline, not the unguarded rewrite.
For each non-holdout prompt: chapter-locked bge RAG, stock Qwen3-4B pass-1
(train2 pass-1 reused when the id and question match; otherwise generated here),
then `../train3/lora_v_rewrite` with the same copy-bias / coverage decoding,
then the span guard plus qwen3:14b fact-diff. Gist-regex applies to must-answer
ids only; these prompts are not the holdout, so gist fallback does not fire.
Protected gist words that already appear in pass-1 are still span-locked.

The SFT input is the single-pass eval format: character card + RAG block + question.
The SFT target is the guarded final answer: the rewrite when it clears span,
fact-diff, and the name checks; otherwise the name-clean pass-1 draft.
A training-time name penalty was not added. The hard filter on targets is the control.

Canonical proper nouns: `data/canonical_names.txt` (174 entries), seeded from `protect.BOOK_NAMES` and extended by scanning
`chapters/*.md` only. `source/*.txt` is the Gutenberg wrapper and was not scanned.
Names added beyond BOOK_NAMES are in `data/canonical_additions.txt`.
1818 spelling variants that appear in the text stay canonical
(Chamounix/Chamouni, De Lacey/De Lacy, Ingolstadt/Ingoldstadt, Werter).
Near-names that are not in the book are rejected. The hard denylist includes
Clervaux, Cliversal, Morritz, Cornelis, Agrippus, Montanvirt, Frankstein.

Teacher generation: eos_token_id [151645, 151643], temperature 0.7, top_p 0.95,
max_new_tokens 200, seed 1 per batch, repetition_penalty 1.15. Pass-1 and the
constrained rewrite ran at batch size 4. Coverage decoding omits
no_repeat_ngram_size so a forced name token cannot be zeroed. The n-gram ban
also stayed off the teacher pass-1, because it stalled batch-1 decode.
605 earlier rows reused train2 pass-1 text when the id and question matched;
the rest are train4 pass-1.

The headline eval does not use the repetition penalty. Smoke on the trained
adapter was 0/20 loops with it and 0/20 loops without it, and the penalty
invented near-names (Agrappa, Morritz, Cliversal, Thoneon, Waldmann). The
no-penalty smoke had none. Official rag_base numbers stay the train1 run.

Train kept 2097 of these 2109 pairs. Twelve were skipped because the chat
template did not share a prefix with the answer (prefix mismatch). The LoRA
is stock Qwen3-4B, r 32, alpha 64, dropout 0.05, lr 2e-5, CE. The 45-minute
reserve stopped it at step 1407/2097 (window-mean CE about 0.83). No CE rollback.

## Counts

- Prompts in the question pool: 2400
- Teacher rows generated (pass-1 + rewrite): 2400
- Fact-diff rows: 2400 (unchecked teacher rows, treated as fallback: 0)
- Pairs kept: 2109
- Targets that are accepted rewrites: 897
- Targets that are pass-1 fallbacks: 1212
- Rewrites rejected for missing protected spans: 95
- Fallbacks whose rewrite missed a span (pair kept as pass-1): 87
- Rewrites rejected for a non-canonical near-name: 71
- Rewrites rejected for dropping a canonical name from pass-1: 272
- Rewrites rejected by fact-diff: 1116
- Pairs dropped for a non-canonical name in pass-1 (no clean target): 83
- Pairs dropped for holdout: 0
- Pairs dropped for Walton voice / 'the stranger': 0
- By character: {'victor': 1274, 'creature': 835}
- By target source: {'pass1_fallback': 1212, 'accepted_rewrite': 897}

## Rewrite reject detail

```
{
 "near_name": 71,
 "factdiff": 1116,
 "missing_canonical": 272,
 "final_no_first_person": 31,
 "missing_span": 95,
 "final_wrong_person": 2,
 "final_frame": 1
}
```

## Pass-1 reject detail (pair dropped)

```
{
 "near_name": 83,
 "no_first_person": 196,
 "wrong_person": 10,
 "spoiler": 2
}
```

## Canonical list

174 names in `data/canonical_names.txt`. Seeded from `protect.BOOK_NAMES` and
extended by a chapters-only scan (`chapters/*.md`). `source/*.txt` is the
Gutenberg wrapper and was not used. Additions are in
`data/canonical_additions.txt` (112 lines). A few multi-word phrases in that
file are recurring adjacent capitals (for example "Gravesend Woolwich");
question templates do not ask those phrases unless they are in `BOOK_NAMES`.
