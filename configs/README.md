# Configs

- `phase1/expN/hparams.json`: run settings for each experiment, the same file the scripts in `src/phase1/expN/` read.
- `phase1/lora/*/adapter_config.json` and `train_report.json`: LoRA hyper-parameters and training summaries for the three adapters behind the locked *No Return* model (facts, pass-1 voice, light DPO rewrite). No weights are included.
- `browser-demo/mlc-config.json`: the MLC config used to quantize the browser model (q4f16_1).
