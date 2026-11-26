# How the training data was made

Hand-written seeds stay in `data/handwritten/` (copies of `../voice/*.jsonl`) and in `../voice/`.
They are never overwritten. Generated files live in `data/generated/`.

## Teacher

- Local Ollama `qwen3:14b`, JSON mode, think off, temperature 0.9.
- Grounding: the chapter file for that unit, truncated if long. Later units are not in the prompt.
- Every row has `reader_pos` and `unit_id`. A line is illegal before that position.

## Held out of all training files

- 34 must-answer questions in `../questions/must-answer.json`
- 18 leak/contamination probes in `../questions/leak-probes.json`
- Drop if the user line is an exact/substring match or Jaccard ≥ 0.62 on normalized tokens.
- Drop if the answer hits a leak-probe fail phrase whose spoiler_pos is still ahead, or 1831 contamination.

## Generated sets

- `generated/facts.jsonl`: 8 factual Q&A per story unit (preface skipped), narrator char.
- `generated/voice_victor.jsonl` / `voice_creature.jsonl`: 6 in-character lines per unit after chat unlock
  (Victor from pos 4 / Letter IV, creature from pos 13 / Vol II Ch II).
- `generated/voice_paraphrase.jsonl`: 3 reworded user lines per handwritten seed, same answer facts.
- `generated/dpo_spoilers.jsonl`: refusal vs leak pairs for light DPO.

## Assembled

- facts.jsonl: 245 rows (generated facts + handwritten seeds reused as fact pairs)
- voice.jsonl: 379 rows (36 handwritten, 232 generated, 111 paraphrase)
- dpo.jsonl: 100 rows
- dropped holdout 1, dropped leak 0

## Voice LoRA choice

One voice LoRA with a character tag in the system prompt (Victor vs creature cards).
Phase 1 used one LoRA for Wren/Diane/YD-4408 the same way. Two adapters would double train
and eval time inside the 8h cap and would complicate the single-pass merged browser candidate.
The card in the prompt is how the model is told who is speaking.
