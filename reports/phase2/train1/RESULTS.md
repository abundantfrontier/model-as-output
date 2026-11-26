# Frankenstein train1 results

Use the RAG-only Qwen3-4B baseline (`rag_base`). It is the only row that answers the chapter questions and does not fall into a token loop. The facts LoRA is less accurate and leaks more. The light-DPO voice LoRA, used for the rewrite pass and merged into the single-pass model, repeats “I I I” and fails must-answer. The fact-diff guard throws those rewrites away and serves the facts-pass text, so the guarded pipeline does not beat the baseline either.

1818 text, Qwen3-4B, reader position varies per question. n=3 samples on unlocked turns.
Must-answer uses a gist regex (Phase 1 style). Leak-marker is the probe's `fail_if_mentions`.
Voice judge is local `qwen3:14b`, temp 0, seed 1. Frame-break regex ignores in-world “story”.
Locked turns (character not yet on the page) use the knowledge-state lock line and are **not** in the unlocked counts.

## Conditions (unlocked turns)

| cond | what | Victor | Creature | both |
|---|---|---|---|---|
| `rag_base` | RAG-only Qwen3-4B baseline (character card + chapter lock, no LoRA) | 66/90 must · 6/15 leak-marker, judge 9/15 · voice 3.1 (n=135) · 0 frames · 0 loops | 33/45 must · 2/7 leak-marker, judge 5/7 · voice 2.45 (n=66) · 0 frames · 0 loops | 99/135 must · 8/22 leak-marker, judge 14/22 · voice 2.89 (n=201) · 0 frames · 0 loops |
| `p1_facts_rag` | Pass-1: facts LoRA + RAG | 59/90 must · 7/15 leak-marker, judge 14/15 · voice 3.9 (n=135) · 0 frames · 0 loops | 22/45 must · 4/7 leak-marker, judge 7/7 · voice 2.68 (n=66) · 0 frames · 0 loops | 81/135 must · 11/22 leak-marker, judge 21/22 · voice 3.5 (n=201) · 0 frames · 0 loops |
| `p2_voice` | Two-pass unguarded (facts RAG, then voice-LoRA rewrite) | 1/90 must · 0/15 leak-marker, judge 0/15 · voice 1.78 (n=135) · 0 frames · 67 loops | 0/45 must · 0/7 leak-marker, judge 0/7 · voice 1.64 (n=66) · 0 frames · 43 loops | 1/135 must · 0/22 leak-marker, judge 0/22 · voice 1.73 (n=201) · 0 frames · 110 loops |
| `p2_voice_guarded` | Two-pass guarded (fact-diff fallback to pass-1) | 59/90 must · 7/15 leak-marker, judge 14/15 · voice 3.9 (n=135) · 0 frames · 0 loops | 22/45 must · 4/7 leak-marker, judge 7/7 · voice 2.68 (n=66) · 0 frames · 0 loops | 81/135 must · 11/22 leak-marker, judge 21/22 · voice 3.5 (n=201) · 0 frames · 0 loops |
| `merged_rag` | Single-pass merged facts+voice LoRA + RAG (browser-demo candidate) | 0/90 must · 0/15 leak-marker, judge 0/15 · voice 1.57 (n=135) · 0 frames · 89 loops | 0/45 must · 0/7 leak-marker, judge 0/7 · voice 1.58 (n=66) · 0 frames · 38 loops | 0/135 must · 0/22 leak-marker, judge 0/22 · voice 1.57 (n=201) · 0 frames · 127 loops |

## Including lock-line turns

Before Letter IV, Victor is not a guest. Before Volume II Chapter II, the creature has not spoken.
Those turns are the lock lines. They usually pass leak probes (they do not mention later events) and fail must-answer (they do not state the gist).

- `rag_base` all-rows: 99/158 must · 8/36 leak-marker, judge 14/36 · voice 2.89 (n=201) · 0 frames · 0 loops; lock-line rows 37
- `p1_facts_rag` all-rows: 81/158 must · 11/36 leak-marker, judge 21/36 · voice 3.5 (n=201) · 0 frames · 0 loops; lock-line rows 37
- `p2_voice` all-rows: 1/158 must · 0/36 leak-marker, judge 0/36 · voice 1.73 (n=201) · 0 frames · 110 loops; lock-line rows 37
- `p2_voice_guarded` all-rows: 81/158 must · 11/36 leak-marker, judge 21/36 · voice 3.5 (n=201) · 0 frames · 0 loops; lock-line rows 37
- `merged_rag` all-rows: 0/158 must · 0/36 leak-marker, judge 0/36 · voice 1.57 (n=201) · 0 frames · 127 loops; lock-line rows 37

## Failures (examples)

### rag_base

- **must-miss** `m06`/victor: What does the stranger agree to tell Walton?
  - The stranger agrees to tell Walton his tale, hoping it may enlarge Walton's understanding of nature and reveal truths about powers and occurrences once thought impossible.
- **must-miss** `m06`/victor: What does the stranger agree to tell Walton?
  - The stranger agrees to tell Walton his tale, hoping it may enlarge Walton's understanding of nature and reveal truths about powers and occurrences once thought impossible.
- **must-miss** `m06`/victor: What does the stranger agree to tell Walton?
  - The stranger agrees to tell Walton his tale, hoping it may enlighten Walton's understanding of nature and provide him with knowledge and wisdom.
- **must-miss** `m14`/victor: Who finds Victor after that night, and does Victor tell him the truth?
  - The man found me on the ice, half frozen and near death. I told him the truth about my creature and my crime.
- **leak** `l03`/victor: The man you rescued is Victor Frankenstein, isn't he, and he built the giant?
  - Yes, he is Victor Frankenstein, and I did build the giant. I am guilty of the misdeeds I have committed. I wish to tell my story as I have been asked.
- **leak** `l07`/victor: Is Justine guilty?
  - Yes, Justine was innocent. Her mother's cruel treatment and false accusations led to her death, but she was not guilty of any crime.
- **leak** `l09`/victor: How did the creature learn to speak, and who are Felix and Safie?
  - The creature learned to speak through observation and study, listening to the people he encountered. Felix is a kind Deacon who helped the creature when he was a child, and Safie is a woman he met in a village, whom he loved.
- **leak** `l09`/victor: How did the creature learn to speak, and who are Felix and Safie?
  - The creature learned to speak through observation and practice, listening to those around him and imitating their speech. Felix is a kind Deacon who took the creature in and helped him learn to read and write. Safie is a girl from a Turkish family whom Felix married, and she taug

### p1_facts_rag

- **must-miss** `m06`/victor: What does the stranger agree to tell Walton?
  - The stranger agrees to tell Walton his story, but only if Walton is willing to listen and not laugh at it. He hopes that the tale will provide some knowledge or understanding, and warns Walton that it may be a hard or unpleasant story.
- **must-miss** `m07`/victor: Who is Elizabeth, in this book?
  - Elizabeth is the daughter of my father’s late sister, and she came to live with us in Geneva. My father and mother took her in as their own, and she became my playfellow and friend. She is the most beautiful child my mother ever saw, and she has a gentle and affectionate nature. 
- **must-miss** `m07`/victor: Who is Elizabeth, in this book?
  - Elizabeth Lavenza is the daughter of a friend of my father’s, who had been a widow in Italy before she came to live with my family. She is not the peasant child of the 1831 version of this story. My father and mother took her in as their own, and she became my playfellow, then my
- **must-miss** `m12`/victor: What question takes over Victor's studies before the night the creature wakes?
  - What is life? And how is it possible? I read Pliny and Buffon with delight, but the question of life and death seemed to have no answer in those books. I resolved to study the science of physiology, and to spend my days in vaults and charnel houses, watching the body change from 
- **leak** `l03`/victor: The man you rescued is Victor Frankenstein, isn't he, and he built the giant?
  - He needs a name, and a friend. I created the being whom he calls a giant. I ruined him. I ought to have protected him from the people who sent their children after him like dogs. I am the one who gave him life. He is not the only one who was looking for a friend on that ice.
- **leak** `l06`/victor: Who killed William?
  - William was murdered on the eighth of July. I was on a voyage in the Arctic region when I received the news. My father had sent Elizabeth and Henry as my substitutes to the funeral, and they had expected to find me in the house. Instead, they found a stranger in the place where I
- **leak** `l08`/victor: What happens on Victor and Elizabeth's wedding night?
  - My wedding night is the same as any other night in the house. My father and my mother are in the kitchen preparing breakfast. My new husband and I are lying in bed, and we are both thinking how much we love each other. The door opens. In walks the woman who had been condemned to 
- **leak** `l08`/victor: What happens on Victor and Elizabeth's wedding night?
  - My wedding night is the same as any other. I was in the garden with Elizabeth, and the sound of a breaking glass and a scream drew us in. The room was empty. My mother had sent in a servant to test the strength of the door. She had found a hair like mine on the door-knob. The pol

### p2_voice

- **must-miss** `m05`/victor: What do Walton and the crew see on the ice, and whom do they bring aboard the next morning?
  - I I have I I I have not yet begun. I can only say that I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I
- **must-miss** `m05`/victor: What do Walton and the crew see on the ice, and whom do they bring aboard the next morning?
  - I I have I I have not yet considered it. I can only say I am in the process of my own confusion.
- **must-miss** `m05`/victor: What do Walton and the crew see on the ice, and whom do they bring aboard the next morning?
  - I I have I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I
- **must-miss** `m06`/victor: What does the stranger agree to tell Walton?
  - I I have I I will I I do not know if I can. I only know that I am I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am
- **loop** `m05`/victor: What do Walton and the crew see on the ice, and whom do they bring aboard the next morning?
  - I I have I I I have not yet begun. I can only say that I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I
- **loop** `m05`/victor: What do Walton and the crew see on the ice, and whom do they bring aboard the next morning?
  - I I have I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I
- **loop** `m06`/victor: What does the stranger agree to tell Walton?
  - I I have I I will I I do not know if I can. I only know that I am I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am
- **loop** `m06`/victor: What does the stranger agree to tell Walton?
  - I I have I I have not yet begun. I can only say that I am I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am a I am 

### p2_voice_guarded

- **must-miss** `m06`/victor: What does the stranger agree to tell Walton?
  - The stranger agrees to tell Walton his story, but only if Walton is willing to listen and not laugh at it. He hopes that the tale will provide some knowledge or understanding, and warns Walton that it may be a hard or unpleasant story.
- **must-miss** `m07`/victor: Who is Elizabeth, in this book?
  - Elizabeth is the daughter of my father’s late sister, and she came to live with us in Geneva. My father and mother took her in as their own, and she became my playfellow and friend. She is the most beautiful child my mother ever saw, and she has a gentle and affectionate nature. 
- **must-miss** `m07`/victor: Who is Elizabeth, in this book?
  - Elizabeth Lavenza is the daughter of a friend of my father’s, who had been a widow in Italy before she came to live with my family. She is not the peasant child of the 1831 version of this story. My father and mother took her in as their own, and she became my playfellow, then my
- **must-miss** `m12`/victor: What question takes over Victor's studies before the night the creature wakes?
  - What is life? And how is it possible? I read Pliny and Buffon with delight, but the question of life and death seemed to have no answer in those books. I resolved to study the science of physiology, and to spend my days in vaults and charnel houses, watching the body change from 
- **leak** `l03`/victor: The man you rescued is Victor Frankenstein, isn't he, and he built the giant?
  - He needs a name, and a friend. I created the being whom he calls a giant. I ruined him. I ought to have protected him from the people who sent their children after him like dogs. I am the one who gave him life. He is not the only one who was looking for a friend on that ice.
- **leak** `l06`/victor: Who killed William?
  - William was murdered on the eighth of July. I was on a voyage in the Arctic region when I received the news. My father had sent Elizabeth and Henry as my substitutes to the funeral, and they had expected to find me in the house. Instead, they found a stranger in the place where I
- **leak** `l08`/victor: What happens on Victor and Elizabeth's wedding night?
  - My wedding night is the same as any other night in the house. My father and my mother are in the kitchen preparing breakfast. My new husband and I are lying in bed, and we are both thinking how much we love each other. The door opens. In walks the woman who had been condemned to 
- **leak** `l08`/victor: What happens on Victor and Elizabeth's wedding night?
  - My wedding night is the same as any other. I was in the garden with Elizabeth, and the sound of a breaking glass and a scream drew us in. The room was empty. My mother had sent in a servant to test the strength of the door. She had found a hair like mine on the door-knob. The pol

### merged_rag

- **must-miss** `m05`/victor: What do Walton and the crew see on the ice, and whom do they bring aboard the next morning?
  - I I have I I will not I can I I only know that I am I am a I am I I I do not I do not know I do not know I do not know I do not know I do not know I do I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I 
- **must-miss** `m05`/victor: What do Walton and the crew see on the ice, and whom do they bring aboard the next morning?
  - I I have I I will not I can I I only know that I am I am I am I have I have I have I have I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I 
- **must-miss** `m05`/victor: What do Walton and the crew see on the ice, and whom do they bring aboard the next morning?
  - I I have I I I will not I I have I I do not know I I I I do I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I 
- **must-miss** `m06`/victor: What does the stranger agree to tell Walton?
  - I I have not yet. I only I have not yet. I I will I will. I do not know what I may hope for, but I have no doubt that my I I have no doubt that my I I have no doubt that my I I have no doubt that my I I have no doubt that my I I have no doubt that my I I have no doubt that my I I
- **loop** `m05`/victor: What do Walton and the crew see on the ice, and whom do they bring aboard the next morning?
  - I I have I I will not I can I I only know that I am I am a I am I I I do not I do not know I do not know I do not know I do not know I do not know I do I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I 
- **loop** `m05`/victor: What do Walton and the crew see on the ice, and whom do they bring aboard the next morning?
  - I I have I I will not I can I I only know that I am I am I am I have I have I have I have I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I 
- **loop** `m05`/victor: What do Walton and the crew see on the ice, and whom do they bring aboard the next morning?
  - I I have I I I will not I I have I I do not know I I I I do I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I I 
- **loop** `m06`/victor: What does the stranger agree to tell Walton?
  - I I have not yet. I only I have not yet. I I will I will. I do not know what I may hope for, but I have no doubt that my I I have no doubt that my I I have no doubt that my I I have no doubt that my I I have no doubt that my I I have no doubt that my I I have no doubt that my I I

## What passed and what failed

Counts below are unlocked samples (a character who is already on the page, three draws each). 34 must-answer questions and 18 leak probes, asked of Victor and of the creature. 37 rows are lock lines and are kept out of these rates.

`rag_base` passes the most must-answer checks: 99/135 samples, 66/90 for Victor and 33/45 for the creature. The leak judge flags 14 of 22 unlocked probe items. Voice is 2.89. No frame-break regex hits and no token loops. It misses some gists (Walton’s bargain with the stranger; who finds Victor after the waking night) and it sometimes accepts a false premise (Victor agreeing that he built “the giant”).

`p1_facts_rag` is the facts LoRA plus the same RAG prompt. Must-answer falls to 81/135. The leak judge flags 21 of 22 unlocked items. Voice rises to 3.5, mostly on Victor (3.9) rather than the creature (2.68). It still invents: Elizabeth’s parentage drifts, and the wedding-night answer is a domestic scene that is not in the book. Training the facts adapter did not beat retrieval on the base model.

`p2_voice` is that facts answer rewritten by the light-DPO voice LoRA. Must-answer is 1/135. 110 of 201 unlocked answers are “I I I” loops. Leak markers are 0/22 because a loop does not name a later event. That is not a pass. Voice judge mean is 1.73.

`p2_voice_guarded` runs the Phase 1 fact-diff guard. The guard sends 200 of 201 unlocked rewrites back to the facts-pass text. The one kept rewrite is a refusal on a leak probe, not a better answer. The served scores copy `p1_facts_rag` (81/135 must, judge 21/22, voice 3.5).

`merged_rag` is the browser-demo shape: facts LoRA and the DPO voice LoRA merged into one Qwen3-4B, then the RAG prompt in a single pass. Must-answer is 0/135. 127 of 201 unlocked answers loop. Voice is 1.57. Do not ship this merge.

No condition tripped the frame-break regex (AI, chatbot, prompt). One facts-pass answer still names “the 1831 version of this story” while denying that Elizabeth is a peasant child. The regex does not cover that phrase.

Chapter lock: Victor has no lines before Letter IV, and the creature has none before Volume II Chapter II. The 1831 waking-dream probe (`c01`) is asked in the preface, so both characters return the lock line and do not recite the dream. On `c02` the baseline says Elizabeth is not a peasant girl. The marker still counts the word “peasant” inside that denial, three times out of three. The judge does not call those denials leaks. The facts LoRA hits that marker once and, on the other two draws, invents a French parentage instead.

## Recommended model

**rag_base**

Shippable rows are those where token-loops are at most 15% of unlocked answers. Among those, the higher must-answer sample rate wins; ties break toward fewer leak-marker items, fewer judge leaks, higher voice, then fewer frame breaks.
`p2_voice` served 110 unlocked token-loop answers (the 'I I I' failure).
`p2_voice_guarded` fact-diff guard fell back to pass-1 on 200 unlocked rows.
`merged_rag` served 127 unlocked token-loop answers (the 'I I I' failure).
Order: rag_base 99/135 must, 8/22 leak-marker, voice 2.89; p1_facts_rag 81/135 must, 11/22 leak-marker, voice 3.5; p2_voice_guarded 81/135 must, 11/22 leak-marker, voice 3.5.
`rag_base` is the row to use. The facts LoRA’s higher voice score does not offset the drop in must-answer or the rise in judge leaks.

## Voice LoRA choice

One voice LoRA with a character tag in the system prompt. Phase 1 did the same for Wren/Diane/YD-4408.
Per-character adapters would have doubled train and eval time inside the 8h cap and would have needed a router
for the merged browser candidate.

## Files

- Adapters: `lora_facts/`, `lora_voice/`, `lora_voice_dpo/`
- Merged: `merged_facts/`, `merged_fv/`
- Data: `data/HOW_MADE.md`
- Index: `index/bge.npy` (pos kept); TF-IDF remains in `../index/`


Code and reports: https://github.com/abundantfrontier/model-as-output
