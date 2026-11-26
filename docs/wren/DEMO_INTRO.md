# Demo intro (No Return — Wren)

*No Return* is a short invented story. Wren is the night-watch officer on the ship Meridian. This page lets you talk with Wren in your own browser. Pick how far you have read, load the model, and ask about the story so far. Wren's answers, and the passages shown under them, use only the chapters you have already opened. Later chapters are not fetched and are not put in the prompt.

The model runs on your computer, through WebLLM and WebGPU. The first load downloads about 2.2 GB of weights and needs about 3.5 GB of GPU memory. Nothing is sent to a server model. This browser build is a single pass: it is the pass-1 voice model with a chapter-locked search. It does not run the second-pass rewrite or the fact guard used in the locked research model.

That locked two-pass model scored 59/60 on story questions, 1/48 on spoiler probes, voice 4.22, and 3 frame breaks. The file this page loads has not been scored after it was quantized for the browser, and WebGPU generation was not part of that build. Wren may still say something the story never said, or may lean later than the chapter you picked. Treat the chat as a way into the story, not as a substitute for reading it.

The weights load from <https://huggingface.co/AbundantFrontierInstitute/wren-no-return-qwen3-4b-q4f16_1-MLC>. Code and reports: <https://github.com/abundantfrontier/model-as-output>.
