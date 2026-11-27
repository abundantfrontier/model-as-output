#!/bin/bash
# Train4 helper. Do not run it from train5: the paths write into train4.
echo "refusing: train5 uses scripts/run_pipeline.sh" >&2
exit 1
