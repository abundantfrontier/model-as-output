# Roadmap

A short public summary of where the project is going. Dates are targets, not promises.

1. **Public demo.** Put Wren from *No Return* on the AFI Lab as an in-browser chat (WebLLM on WebGPU). The weights will be hosted on Hugging Face at `https://huggingface.co/HF_ORG/wren-qwen3-4b-q4f16_1-MLC` (coming soon, link TBD). Before launch, spot-check the quantized browser file on WebGPU.
2. **Phase 2 voice.** Fix the *Frankenstein* voice adapter with one bounded retrain that follows the Phase 1 rewrite recipe. train2 fixed the loops but scored 83/135 against the baseline's 99/135, so the next step is a rewrite that keeps names and key words. If that still falls short, try one recipe that works across books (voice from a few examples plus retrieval) rather than tuning each book.
3. **Fidelity and citation.** Treat the model as an interface over a frozen, citable text. Every answer links to the passages it came from, and reviewers sign off on the frozen text plus the evaluation suite.
4. **Versioning and provenance.** Tie model versions to text versions. The 1818 vs 1831 *Frankenstein* checks are a first example. Also cover multi-author provenance and archival durability (a model that still runs offline years later).
5. **Phase 3.** Turn a real research paper or thesis into a model, which is the original "model as output" idea.
