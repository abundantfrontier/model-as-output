# Data

Small, hand-made data only. Generated training sets and indexes are not committed; the scripts rebuild them.

- `no-return/`: what each character knows and when (`knowledge-state.yaml`) and the spoiler (leak) probes for *No Return*. The Phase 1 must-answer questions live in `src/phase1/exp*/common.py`.
- `frankenstein/`: the 34 must-answer questions, 16 leak probes plus 2 checks for 1831-edition contamination, the knowledge state (chat unlocks: Victor from Letter IV, the creature from Volume II, Chapter II), and hand-written voice seeds for Victor and the creature.
  - The novel itself is **not** included. Download Project Gutenberg #41445 (the 1818 text); `frankenstein/SOURCE.md` has the URL and SHA-256.
