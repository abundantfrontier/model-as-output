/* Separate WebLLM config for the Hugging Face-hosted Wren model.
 * Use this instead of the local ./models/ paths in app.js when you want
 * the browser to download weights from Hugging Face.
 * Local app.js is unchanged and still serves the on-disk build.
 */
export const WREN_HF_MODEL = {
  model: "https://huggingface.co/AbundantFrontierInstitute/wren-no-return-qwen3-4b-q4f16_1-MLC",
  model_id: "wren-q4f16_1",
  model_lib: "https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_87/base/Qwen3-4B-q4f16_1_cs1k-webgpu.wasm",
  overrides: { context_window_size: 4096, prefill_chunk_size: 1024 },
  vram_required_MB: 3500,
};
