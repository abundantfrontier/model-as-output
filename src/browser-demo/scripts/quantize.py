#!/usr/bin/env python3
"""Quantize the merged Wren weights to MLC q4f16_1 on CPU.

Nightly mlc_llm passes a TVM DataType into numpy.ndarray.astype. NumPy 2
rejects that object, and NumPy 2.5 also rejects replacing ndarray.astype.
This process rewrites the four Qwen3 loader casts to np.dtype(str(dtype)).
The installed package is not edited.

CUDA_VISIBLE_DEVICES is cleared before mlc_llm imports torch, and the
convert call passes --device cpu. This job must not allocate the 3090.

After the weights land, the same process writes mlc-chat-config.json with
the qwen3 chat template, context 4096, and prefill chunk 1024 so it matches
the prebuilt Qwen3-4B-q4f16_1_cs1k WebGPU library.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import inspect

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "work" / "merged-voice"
MLC_CONFIG = ROOT / "work" / "mlc-config.json"
MLC_SRC = ROOT / "work" / "mlc-src"
OUT = ROOT / "dist" / "models" / "wren-q4f16_1-MLC"


def _np_dtype(dtype):
    try:
        return np.dtype(dtype)
    except (TypeError, ValueError):
        return np.dtype(str(dtype))


def install_cast_patch():
    """Retarget Qwen3's weight lambdas so they cast with a NumPy dtype.

    NumPy 2.5 rejects assigning ndarray.astype, and it rejects a TVM DataType
    object. The four casts in qwen3_loader are rewritten in this process only.
    """
    import mlc_llm.model.qwen3.qwen3_loader as loader
    from mlc_llm.model.model import MODELS

    src = inspect.getsource(loader.huggingface)
    old = ".astype(dtype)"
    new = ".astype(_np_dtype(dtype))"
    if old not in src:
        raise SystemExit("qwen3 loader has no .astype(dtype) to patch")
    src = src.replace(old, new)
    loader._np_dtype = _np_dtype
    exec(src, loader.__dict__)
    patched = loader.huggingface
    for key in ("huggingface-torch", "huggingface-safetensor"):
        MODELS["qwen3"].source[key] = patched
    return loader


def _round_half_away(values: np.ndarray) -> np.ndarray:
    offset = np.where(values >= 0, 0.5, -0.5)
    return np.trunc(values + offset)


def numpy_quantize_weight(self, weight, axis: int = -1, output_transpose: bool = False):
    """Group-q4 pack in NumPy.

    The CPU nightly segfaults while compiling the TVM quantize kernel for the
    embedding matrix (151936, 2560). q4f16_1 uses layout NK, so the packed
    axis is the last one and nothing is transposed. Nibbles are little-endian
    in each uint32, and the stored code is round(weight / scale) + 7.
    """
    import tvm

    if output_transpose:
        raise RuntimeError("q4f16_1 NK layout does not transpose packed weights")
    values = np.ascontiguousarray(weight.numpy())
    if values.dtype != np.float16:
        values = values.astype(np.float16)
    if axis < 0:
        axis += values.ndim
    group = int(self.group_size)
    max_int = int(self.max_int_value)
    per_storage = int(self.num_elem_per_storage)
    width = values.shape[axis]
    if width % group != 0:
        raise RuntimeError(f"group axis {width} is not a multiple of {group}")
    moved = np.moveaxis(values, axis, -1)
    grouped = moved.reshape(*moved.shape[:-1], width // group, group)
    max_abs = np.max(np.abs(grouped), axis=-1)
    scale = (max_abs / np.float16(max_int)).astype(np.float16)
    safe = scale.astype(np.float32)
    safe = np.where(safe == 0, np.float32(1), safe)
    coded = _round_half_away(
        grouped.astype(np.float32) / safe[..., None] + np.float32(max_int)
    )
    coded = np.where(scale[..., None] == 0, np.float32(max_int), coded)
    coded = np.clip(coded, 0, max_int * 2).astype(np.uint32)
    coded = coded.reshape(*moved.shape)
    coded = coded.reshape(*coded.shape[:-1], width // per_storage, per_storage)
    shifts = (np.arange(per_storage, dtype=np.uint32) * np.uint32(4))
    packed = np.sum(coded << shifts, axis=-1, dtype=np.uint32)
    packed = np.moveaxis(packed, -1, axis)
    scale = np.moveaxis(scale, -1, axis)
    device = weight.device
    return [
        tvm.runtime.tensor(np.ascontiguousarray(packed), device=device),
        tvm.runtime.tensor(np.ascontiguousarray(scale), device=device),
    ]


def install_quant_patch() -> None:
    from mlc_llm.quantization.group_quantization import GroupQuantize

    GroupQuantize.quantize_weight = numpy_quantize_weight
    print("QUANT_PATCH numpy group-q4", flush=True)


def _unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    shifts = (np.arange(8, dtype=np.uint32) * np.uint32(4))
    return ((packed[..., None] >> shifts) & np.uint32(0xF)).astype(np.int32)


def check_quant() -> None:
    """Check the NumPy pack against the q4f16_1 dequant formula.

    The CPU nightly segfaults inside relax.build for every group-quant
    kernel, including a 64x128 one, so there is no TVM reference on this
    machine. This checks shapes, the zero code, and reconstruction error.
    """
    import tvm
    from mlc_llm.quantization.quantization import QUANTIZATION

    spec = QUANTIZATION["q4f16_1"]
    device = tvm.runtime.device("cpu", 0)

    zeros = np.zeros((2, 32), dtype=np.float16)
    packed, scale = numpy_quantize_weight(
        spec, tvm.runtime.tensor(zeros, device=device)
    )
    if packed.numpy().shape != (2, 4) or scale.numpy().shape != (2, 1):
        raise SystemExit(f"zero shapes {packed.numpy().shape} {scale.numpy().shape}")
    if int(packed.numpy().reshape(-1)[0]) != 0x77777777:
        raise SystemExit(f"zero code packed as {packed.numpy().reshape(-1)[0]:#x}")
    if float(scale.numpy().reshape(-1)[0]) != 0.0:
        raise SystemExit("zero group scale is not 0")

    ones = np.full((1, 32), 7.0, dtype=np.float16)
    packed, scale = numpy_quantize_weight(
        spec, tvm.runtime.tensor(ones, device=device)
    )
    if int(packed.numpy().reshape(-1)[0]) != 0xEEEEEEEE:
        raise SystemExit(f"const-7 packed as {packed.numpy().reshape(-1)[0]:#x}")
    if float(scale.numpy().reshape(-1)[0]) != 1.0:
        raise SystemExit(f"const-7 scale {float(scale.numpy().reshape(-1)[0])}")

    spike = np.zeros((1, 32), dtype=np.float16)
    spike[0, 0] = np.float16(7)
    packed, scale = numpy_quantize_weight(
        spec, tvm.runtime.tensor(spike, device=device)
    )
    word = int(packed.numpy().reshape(-1)[0])
    # First element code 14 sits in the low nibble. The rest are the zero code 7.
    if word != 0x7777777E:
        raise SystemExit(f"spike packed as {word:#x}, expected 0x7777777e")
    if float(scale.numpy().reshape(-1)[0]) != 1.0:
        raise SystemExit(f"spike scale {float(scale.numpy().reshape(-1)[0])}")

    rng = np.random.default_rng(1)
    raw = rng.normal(0, 0.08, size=(64, 128)).astype(np.float16)
    raw[0, :32] = 0
    packed, scale = numpy_quantize_weight(
        spec, tvm.runtime.tensor(np.ascontiguousarray(raw), device=device)
    )
    codes = _unpack_nibbles(packed.numpy()).reshape(raw.shape)
    scale_full = np.repeat(scale.numpy().astype(np.float32), spec.group_size, axis=-1)
    recon = (codes - 7).astype(np.float32) * scale_full
    err = np.max(np.abs(recon - raw.astype(np.float32)))
    limit = float(np.max(np.abs(raw.astype(np.float32))) / 14.0 + 1e-3)
    print(f"CHECK_QUANT recon_max_abs {err:.6f} limit {limit:.6f}", flush=True)
    if err > limit:
        raise SystemExit("reconstruction error is above one q4 bin")
    print("CHECK_QUANT_OK", flush=True)


def probe() -> None:
    import torch
    import tvm

    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible; refusing to run")
    loader = install_cast_patch()
    from mlc_llm.model.model import MODELS

    if MODELS["qwen3"].source["huggingface-safetensor"] is not loader.huggingface:
        raise SystemExit("model registry still points at the unpatched loader")
    names = set()
    def walk(code):
        names.update(code.co_names)
        for const in code.co_consts:
            if hasattr(const, "co_names"):
                walk(const)
    walk(loader.huggingface.__code__)
    if "_np_dtype" not in names:
        raise SystemExit(f"patched loader missing _np_dtype: {sorted(names)}")
    dt = tvm.runtime.DataType("float16")
    base = np.ones((4, 8), dtype=np.float32)
    cast = base.astype(_np_dtype(dt))
    if cast.dtype != np.float16:
        raise SystemExit(f"astype produced {cast.dtype}")
    fused = np.concatenate([base, base], axis=0).astype(_np_dtype(dt))
    if fused.shape != (8, 8) or fused.dtype != np.float16:
        raise SystemExit(f"concat astype failed: {fused.shape} {fused.dtype}")
    tensor = tvm.runtime.tensor(cast, device=tvm.runtime.device("cpu", 0))
    if str(tensor.dtype) != "float16":
        raise SystemExit(f"tvm tensor dtype {tensor.dtype}")
    print("PROBE_OK", cast.dtype, tuple(tensor.shape), str(tensor.dtype), flush=True)


def prepare_src() -> Path:
    if not MLC_CONFIG.is_file():
        raise SystemExit(f"missing {MLC_CONFIG}")
    MLC_SRC.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MLC_CONFIG, MLC_SRC / "config.json")
    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        src = MERGED / name
        if not src.is_file():
            raise SystemExit(f"missing {src}")
        dest = MLC_SRC / name
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(src)
    return MLC_SRC / "config.json"


def run_mlc(argv: list[str]) -> None:
    sys.argv = ["mlc_llm", *argv]
    from mlc_llm.__main__ import main

    main()


def convert() -> None:
    import torch

    if torch.cuda.is_available():
        raise SystemExit("CUDA is visible; refusing to convert")
    install_cast_patch()
    install_quant_patch()
    (ROOT / "work" / "quantize.pid").write_text(str(os.getpid()) + "\n")
    if not (MERGED / "model.safetensors").is_file():
        raise SystemExit(f"missing merged weights under {MERGED}")
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    print("CONVERT_START device=cpu cuda=False", flush=True)
    run_mlc(
        [
            "convert_weight",
            str(MLC_CONFIG),
            "--quantization",
            "q4f16_1",
            "--model-type",
            "qwen3",
            "--device",
            "cpu",
            "--source",
            str(MERGED),
            "--source-format",
            "huggingface-safetensor",
            "--output",
            str(OUT),
        ]
    )
    print("CONVERT_DONE", flush=True)


def gen_config() -> None:
    config = prepare_src()
    print("GEN_CONFIG_START", flush=True)
    run_mlc(
        [
            "gen_config",
            str(config),
            "--quantization",
            "q4f16_1",
            "--model-type",
            "qwen3",
            "--conv-template",
            "qwen3",
            "--context-window-size",
            "4096",
            "--prefill-chunk-size",
            "1024",
            "--output",
            str(OUT),
        ]
    )
    print("GEN_CONFIG_DONE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--check-quant", action="store_true")
    parser.add_argument("--gen-config-only", action="store_true")
    args = parser.parse_args()
    if args.probe:
        probe()
        return
    if args.check_quant:
        check_quant()
        return
    if args.gen_config_only:
        gen_config()
        return
    convert()
    gen_config()


if __name__ == "__main__":
    main()
