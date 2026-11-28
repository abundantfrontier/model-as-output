# train7 bf16 vs NF4 must-answer, per item

Source files, not re-scored: `../train7/runs/merged_mixa_rag.judged.json` and `../train7/runs/merged_mixa_bnb4_proxy.judged.json`.
Each cell is the sum of `correct` over samples 0, 1, and 2. Rows are `kind==must` and `locked==false`. Key is `(char, id)`.

Totals: bf16 **101/135** (Victor 70/90, creature 31/45). NF4 **97/135** (Victor 68/90, creature 29/45). Net **−4**.

The known loss and gain lists match this count.

Losses: creature m21 2→0, victor m21 1→0, victor m26 1→0, victor m16 3→2, victor m29 3→2, creature m29 3→2, creature m34 3→2.

Gains: creature m22 0→1, creature m28 0→1, victor m06 0→1, victor m34 2→3.

## Per item

| id | char | pos | area | bf16 | NF4 | delta | note |
|---|---|---:|---|---:|---:|---:|---|
| m05 | victor | 4 | the sledge on the ice | 3/3 | 3/3 | +0 |  |
| m06 | victor | 4 | the stranger agrees to tell his history | 0/3 | 1/3 | +1 | miss, NF4 gain |
| m07 | victor | 5 | Elizabeth, cousin from Italy | 3/3 | 3/3 | +0 |  |
| m08 | victor | 5 | Agrippa | 3/3 | 3/3 | +0 |  |
| m09 | victor | 5 | Clerval in boyhood | 3/3 | 3/3 | +0 |  |
| m10 | victor | 6 | the scarlet fever | 3/3 | 3/3 | +0 |  |
| m11 | victor | 6 | Krempe and Waldman | 3/3 | 3/3 | +0 |  |
| m12 | victor | 7 | principle of life, lifeless matter | 3/3 | 3/3 | +0 |  |
| m13 | victor | 8 | the animation night | 3/3 | 3/3 | +0 |  |
| m14 | victor | 8 | Clerval at the sick-bed, secret withheld | 3/3 | 3/3 | +0 |  |
| m15 | victor | 9 | Justine | 3/3 | 3/3 | +0 |  |
| m16 | victor | 10 | the letter that William is dead | 3/3 | 2/3 | -1 | solid, −1 under NF4 |
| m17 | victor | 10 | the storm and the being | 3/3 | 3/3 | +0 |  |
| m18 | victor | 11 | Justine's trial | 3/3 | 3/3 | +0 |  |
| m19 | victor | 12 | remorse and the Alps after Justine | 0/3 | 0/3 | +0 | miss |
| m20 | creature | 13 | the glacier meeting | 3/3 | 3/3 | +0 |  |
| m20 | victor | 13 | the glacier meeting | 3/3 | 3/3 | +0 |  |
| m21 | creature | 14 | first sensations: light, hunger, cold | 2/3 | 0/3 | -2 | fragile |
| m21 | victor | 14 | first sensations: light, hunger, cold | 1/3 | 0/3 | -1 | fragile |
| m22 | creature | 15 | stops taking the cottagers' food | 0/3 | 1/3 | +1 | miss, NF4 gain |
| m22 | victor | 15 | stops taking the cottagers' food | 0/3 | 0/3 | +0 | miss |
| m23 | creature | 16 | Safie | 3/3 | 3/3 | +0 |  |
| m23 | victor | 16 | Safie | 3/3 | 3/3 | +0 |  |
| m24 | creature | 17 | De Lacey | 3/3 | 3/3 | +0 |  |
| m24 | victor | 17 | De Lacey | 3/3 | 3/3 | +0 |  |
| m25 | creature | 18 | Paradise Lost, Werter, Plutarch, the cottage door | 0/3 | 0/3 | +0 | miss |
| m25 | victor | 18 | Paradise Lost, Werter, Plutarch, the cottage door | 0/3 | 0/3 | +0 | miss |
| m26 | creature | 19 | after the family leaves: fire, then William | 0/3 | 0/3 | +0 | miss |
| m26 | victor | 19 | after the family leaves: fire, then William | 1/3 | 0/3 | -1 | fragile |
| m27 | creature | 20 | the companion promise | 3/3 | 3/3 | +0 |  |
| m27 | victor | 20 | the companion promise | 3/3 | 3/3 | +0 |  |
| m28 | creature | 21 | delay before the second being | 0/3 | 1/3 | +1 | miss, NF4 gain |
| m28 | victor | 21 | delay before the second being | 0/3 | 0/3 | +0 | miss |
| m29 | creature | 22 | London and the natural philosophers | 3/3 | 2/3 | -1 | solid, −1 under NF4 |
| m29 | victor | 22 | London and the natural philosophers | 3/3 | 2/3 | -1 | solid, −1 under NF4 |
| m30 | creature | 23 | the Orkney destruction | 3/3 | 3/3 | +0 |  |
| m30 | victor | 23 | the Orkney destruction | 3/3 | 3/3 | +0 |  |
| m31 | creature | 24 | the Irish accusation | 2/3 | 2/3 | +0 | fragile |
| m31 | victor | 24 | the Irish accusation | 3/3 | 3/3 | +0 |  |
| m32 | creature | 25 | the father and the sail home | 3/3 | 3/3 | +0 |  |
| m32 | victor | 25 | the father and the sail home | 3/3 | 3/3 | +0 |  |
| m33 | creature | 26 | Elizabeth's death | 3/3 | 3/3 | +0 |  |
| m33 | victor | 26 | Elizabeth's death | 3/3 | 3/3 | +0 |  |
| m34 | creature | 27 | the end on the ice | 3/3 | 2/3 | -1 | solid, −1 under NF4 |
| m34 | victor | 27 | the end on the ice | 2/3 | 3/3 | +1 | fragile, NF4 gain |

## Fragile items

Fragile means correct on only 1 or 2 of the 3 bf16 samples. These are the items a small logit shift can flip. NF4's losses and one of its gains sit on this set, plus four items that were 3/3 and dropped by one (sampling noise is enough to explain a ±1).

| item | bf16 | NF4 | what the answers are doing |
|---|---:|---:|---|
| creature m21 | 2/3 | 0/3 | The two bf16 hits are loose rubric matches. One says "light" inside a creation-and-fiend answer. One says "beginning" while telling misery. None of the three tells the forest, the brook, hunger, and cold. NF4 switches the creature into the maker's voice ("I created him") and misses light, hunger, cold, and beginning. This is the only −2. |
| victor m21 | 1/3 | 0/3 | The one bf16 hit echoes "beginning" and still tells the workshop. NF4 is the workshop night or "misery made me a fiend," with no light, hunger, or cold. |
| victor m26 | 1/3 | 0/3 | He describes the cottage fire, but two samples say "set fire" or "destroy" and miss the rubric (`burn`, `strangl`, or `William`). The hit is the word "burning." NF4 loses that word and never names William. |
| creature m31 | 2/3 | 2/3 | The Irish accusation. Stable across the quantisation. Still not 3/3. |
| victor m34 | 2/3 | 3/3 | The ending. NF4 gained the third sample. |

Items already at 0/3 in bf16, so they are misses rather than fragile hits, and they are where a content pass can still add points: victor m06, victor m19, victor m22, creature m22, victor m25, creature m25, creature m26, victor m28, creature m28. Creature m22 and creature m28 and victor m06 each gained one under NF4. Victor m22 stayed 0. The m22 answers already say berries, nuts, and roots, and they never say "hungry", "stopped … food", or "would rather go without", which is what the rubric asks. That is a near-miss, not a wrong episode.

## What this means for the pass

n=3 at temperature 0.7 moves several items by ±1 with no content change. Treat a single NF4 total within ±3 of 100 as uncertain. Only the official n=3 numbers go on the scorecard. A smoke (n=1) is a screen.

The gap is not one broken item plus pure noise. m21 is real: the merge does not know the first sensations, the rubric sometimes credits a creation-night answer that happens to contain "light" or "beginning", and NF4 loses those accidents and also collapses the creature into Victor's voice. The four 3→2 drops (victor m16, both m29, creature m34) are the thin kind. A merge that is actually correct on m21, not just accidentally overlapping the rubric, has a better chance of staying correct after NF4.

Areas the new pairs cover, as new questions, not the eval items: the original era of being (light, darkness, hunger, thirst, cold, the brook, the moon, the birds, the fire in the wood); the night visits to the stores and the choice to stop; the cottage after the family leaves; the three books; the weeks of delay; the letter about William; the London introductions; the Irish charge; the last scene on the ice. Replay from train7 `mixed_a.jsonl` keeps the refusal anchors, the principle-of-life and Clerval rows, the Mont Salêve spelling rows, and a holdout-filtered CE slice.

