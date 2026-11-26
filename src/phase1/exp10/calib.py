"""Voice-judge calibration: noise grid, second local judge, Spearman on condition means."""
import os, json, math, argparse, subprocess, statistics as st
import frozen
from judge_wrap import score_file, VOICE_RUBRIC

H = frozen.H
CONDS = [
    "p2_light_dpo_guarded",
    "p2_light_dpo",
    "v_merged_v_rag",
    "q4_base_rag",
    "v_fb_dpo_v",
]
SEEDS = (1, 2, 3)
TEMPS = (0.0, 0.3)
QWEN = "qwen3:14b"
GEMMA = "gemma3:12b"
ORIG = {
    "p2_light_dpo_guarded": 4.215686274509804,
    "p2_light_dpo": 4.279411764705882,
    "v_merged_v_rag": 4.200980392156863,
    "q4_base_rag": 3.8872549019607843,
    "v_fb_dpo_v": 4.362745098039215,
}


def ollama_names():
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
    except Exception as e:
        return [], str(e)
    names = []
    for line in out.splitlines()[1:]:
        parts = line.split()
        if parts:
            names.append(parts[0])
    return names, out


def pick_second():
    names, raw = ollama_names()
    if GEMMA in names or any(n.startswith("gemma3:12b") for n in names):
        return GEMMA, raw
    return None, raw


def src_path(cond):
    return os.path.join(H, "calib", "src", cond + ".json")


def cell_out(model, cond, temp, seed):
    tag = model.replace(":", "_")
    d = os.path.join(H, "calib", tag)
    os.makedirs(d, exist_ok=True)
    t = "0" if float(temp) == 0 else str(temp).replace(".", "p")
    return os.path.join(d, f"{cond}.t{t}.s{seed}.voice.json")


def mean_of(path):
    d = json.load(open(path))
    vs = [r["voice_score"] for r in d["results"] if r.get("voice_score")]
    return (sum(vs) / len(vs)) if vs else None, len(vs), len(d["results"])


def spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def run_grid(model, temps, seeds, limit=0, force=False):
    for cond in CONDS:
        src = src_path(cond)
        if not os.path.isfile(src):
            raise SystemExit(f"missing calib src {src}")
        for temp in temps:
            for seed in seeds:
                out = cell_out(model, cond, temp, seed)
                score_file(src, out, model, temp, seed, voice_only=True, force=force, limit=limit)


def noise_table(model=QWEN):
    rows = []
    per = {}
    for cond in CONDS:
        cells = {}
        vals = []
        for temp in TEMPS:
            for seed in SEEDS:
                path = cell_out(model, cond, temp, seed)
                if not os.path.isfile(path):
                    cells[(temp, seed)] = None
                    continue
                m, n_v, n = mean_of(path)
                cells[(temp, seed)] = dict(mean=m, n_voice=n_v, n=n, path=os.path.relpath(path, H))
                if m is not None:
                    vals.append(m)
        mean = st.mean(vals) if vals else None
        std = st.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None)
        t0s1 = cells.get((0.0, 1))
        orig = ORIG.get(cond)
        match = None
        if t0s1 and t0s1["mean"] is not None and orig is not None:
            match = abs(t0s1["mean"] - orig) < 0.02
        per[cond] = dict(
            cells={f"t{t}_s{s}": cells[(t, s)] for t in TEMPS for s in SEEDS},
            mean=mean, std=std, n_cells=len(vals),
            t0_s1=None if not t0s1 else t0s1["mean"],
            original=orig,
            t0_s1_matches_original=match,
        )
        rows.append(per[cond])
    return per


def second_table(model, qwen_t0s1):
    means = {}
    for cond in CONDS:
        path = cell_out(model, cond, 0.0, 1)
        if not os.path.isfile(path):
            means[cond] = None
            continue
        m, n_v, n = mean_of(path)
        means[cond] = dict(mean=m, n_voice=n_v, n=n, path=os.path.relpath(path, H))
    xs, ys = [], []
    for cond in CONDS:
        q = qwen_t0s1.get(cond)
        g = None if means[cond] is None else means[cond]["mean"]
        if q is not None and g is not None:
            xs.append(q)
            ys.append(g)
    rho = spearman(xs, ys) if len(xs) == 5 else None
    return dict(model=model, means=means, spearman_vs_qwen3_14b_t0_s1=rho, n_paired=len(xs))


def write_report():
    noise = noise_table(QWEN)
    qwen_t0s1 = {c: noise[c]["t0_s1"] for c in CONDS}
    second_model, listing = pick_second()
    second = None
    skip_reason = None
    if second_model is None:
        skip_reason = "no suitable second local model (wanted gemma3:12b, similar size to qwen3:14b, already downloaded)"
    else:
        second = second_table(second_model, qwen_t0s1)
    doc = dict(
        rubric=VOICE_RUBRIC,
        conditions=CONDS,
        qwen_noise=noise,
        second_judge=second,
        second_skip_reason=skip_reason,
        ollama_list=listing,
        original_t0_s1=ORIG,
        note="voice-only rescoring of fixed answer files; answers were not regenerated",
    )
    out = os.path.join(H, "calib", "report.json")
    frozen.dump_json(out, doc)
    print("wrote", out)
    print("qwen noise:")
    for cond in CONDS:
        n = noise[cond]
        print(f"  {cond}: mean={n['mean'] and round(n['mean'],4)} std={n['std'] and round(n['std'],4)} t0s1={n['t0_s1'] and round(n['t0_s1'],4)} orig={round(n['original'],4)} match={n['t0_s1_matches_original']}")
    if second:
        print("second", second["model"], "spearman", second["spearman_vs_qwen3_14b_t0_s1"])
        for cond in CONDS:
            m = second["means"][cond]
            print(f"  {cond}: {None if not m else round(m['mean'], 4)}")
    elif skip_reason:
        print("second skip:", skip_reason)
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["noise", "second", "report", "self-test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.cmd == "self-test":
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert abs(spearman(xs, ys) - 1.0) < 1e-9
        ys2 = [5.0, 4.0, 3.0, 2.0, 1.0]
        assert abs(spearman(xs, ys2) + 1.0) < 1e-9
        print("calib self-test ok")
        return
    if a.cmd == "noise":
        run_grid(QWEN, TEMPS, SEEDS, limit=a.limit, force=a.force)
        write_report()
        return
    if a.cmd == "second":
        model, listing = pick_second()
        print("ollama list:\n", listing)
        if model is None:
            print("SKIP second judge: no gemma3:12b (or similar/larger local chat model besides qwen3:14b)")
            write_report()
            return
        print("second judge", model)
        run_grid(model, (0.0,), (1,), limit=a.limit, force=a.force)
        write_report()
        return
    if a.cmd == "report":
        write_report()


if __name__ == "__main__":
    main()
