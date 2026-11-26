# Roadmap

A short public summary of where the project is going. Dates are targets, not promises.

1. **Public demo.** Put Wren from *No Return* on the AFI Lab as an in-browser chat (WebLLM on WebGPU). The weights are hosted on Hugging Face at <https://huggingface.co/AbundantFrontierInstitute/wren-no-return-qwen3-4b-q4f16_1-MLC>. Before launch, spot-check the quantized browser file on WebGPU.
2. **Phase 2 voice.** train3's guarded constrained rewrite beat the retrieval baseline: 100/135 vs 99/135, two-judge voice 3.33, 0 loops. The merged single-pass model (62/135) should not ship. The remaining gap is a single-pass browser model that copies protected names and key words without a two-pass processor. If that still falls short, try one recipe that works across books (voice from a few examples plus retrieval) rather than tuning each book.
3. **Fidelity and citation.** Treat the model as an interface over a frozen, citable text. Every answer links to the passages it came from, and reviewers sign off on the frozen text plus the evaluation suite.
4. **Versioning and provenance.** Tie model versions to text versions. The 1818 vs 1831 *Frankenstein* checks are a first example. Also cover multi-author provenance and archival durability (a model that still runs offline years later).
5. **Phase 3.** Turn a real research paper or thesis into a model, which is the original "model as output" idea.
