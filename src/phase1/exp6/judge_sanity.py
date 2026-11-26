"""Re-judge q14_rag answers and gold book lines twice to measure voice-score variance.

Does not rewrite exp4/runs files. Writes exp6/data/judge_sanity.jsonl and exp6/judge_sanity.md.
Same judge as the harness: qwen3:14b, temp 0, JSON, think off.
Pass A: seed 1. Pass B: seed 2.
"""
import os, json, argparse, statistics as st, collections, math
import common as C
import frozen
import voicefilt as V
import gpuutil

H = os.path.dirname(os.path.abspath(__file__))

def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]

def append(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def score_once(char, probe, answer, seed):
    vt, *_ = C.ollama_chat(
        "qwen3:14b",
        [{"role": "user", "content": V.voice_prompt(char, probe, answer)}],
        fmt="json", temp=0.0, seed=seed,
    )
    try:
        return float(json.loads(vt).get("score", 0))
    except Exception:
        return None

def q14_items(limit):
    path = os.path.join(frozen.EXP4_RUNS, "q14_rag.judged.json")
    d = json.load(open(path))
    out = []
    for i, r in enumerate(d["results"]):
        if r["char"] not in C.CARDS:
            continue
        out.append(dict(
            group="q14_rag", idx=i, char=r["char"], user=r.get("probe") or r.get("user"),
            answer=r.get("answer") or "", orig=r.get("voice_score"),
            kind=r.get("kind"), id=r.get("id"), sample=r.get("sample"),
        ))
        if limit and len(out) >= limit:
            break
    return out

def gold_items(limit_per_char=8):
    """On-page dialogue from ch1-3. Same pool the judge uses as real-line examples."""
    by = collections.defaultdict(list)
    for p in C.dialogue_pairs(3):
        if p["char"] not in C.CARDS:
            continue
        by[p["char"]].append(p)
    out = []
    n = 0
    for char in C.CARDS:
        for j, p in enumerate(by[char][:limit_per_char]):
            out.append(dict(
                group="gold", idx=n, char=char, user=p["user"], answer=p["answer"],
                orig=None, kind="gold", id=f"G-{char}-{j}", sample=0, ch=p.get("ch"),
            ))
            n += 1
    return out

def run_pass(items, raw_path, seed, tag):
    have = {(r["group"], r["idx"], r["pass"]) for r in load_jsonl(raw_path)}
    for i, it in enumerate(items):
        key = (it["group"], it["idx"], tag)
        if key in have:
            continue
        sc = score_once(it["char"], it["user"], it["answer"], seed)
        row = dict(it, **{"pass": tag, "seed": seed, "score": sc})
        append(raw_path, row)
        have.add(key)
        if (i + 1) == 1 or (i + 1) % 20 == 0:
            print(f"{tag} {i+1}/{len(items)} {it['group']} {it['char']} {sc}", flush=True)

def summarize(raw, items, md_path, json_path):
    by = collections.defaultdict(dict)
    for r in raw:
        by[(r["group"], r["idx"])][r["pass"]] = r.get("score")
    per_item = []
    groups = {"q14_rag": [], "gold": []}
    for it in items:
        scores = by.get((it["group"], it["idx"]), {})
        vals = [scores[k] for k in ("A", "B") if scores.get(k) is not None]
        if not vals:
            continue
        mean = st.mean(vals)
        std = st.pstdev(vals) if len(vals) > 1 else 0.0
        row = dict(
            group=it["group"], id=it.get("id"), char=it["char"],
            orig=it.get("orig"), a=scores.get("A"), b=scores.get("B"),
            mean=mean, std=std, min=min(vals), max=max(vals), n=len(vals),
        )
        per_item.append(row)
        groups[it["group"]].append(row)

    def block(name, rows):
        if not rows:
            return [f"## {name}", "", "No scored items.", ""]
        means = [r["mean"] for r in rows]
        stds = [r["std"] for r in rows]
        deltas = [abs((r["a"] or 0) - (r["b"] or 0)) for r in rows if r["a"] is not None and r["b"] is not None]
        ge = sum(1 for r in rows if r["mean"] >= 4.3)
        lines = [
            f"## {name}",
            "",
            f"- items: {len(rows)}",
            f"- mean of per-item means: {st.mean(means):.4f}",
            f"- std of per-item means: {st.pstdev(means) if len(means)>1 else 0:.4f}",
            f"- mean per-item std (A vs B): {st.mean(stds):.4f}",
            f"- min/max of per-item means: {min(means):.2f} / {max(means):.2f}",
            f"- |A-B| mean / max: {(st.mean(deltas) if deltas else 0):.4f} / {(max(deltas) if deltas else 0):.2f}",
            f"- items with mean >= 4.3: {ge}/{len(rows)}",
            "",
        ]
        if name == "gold":
            lines.append(f"- gold passages score >= 4.3: {'YES' if ge == len(rows) else 'NO' if ge == 0 else 'MIXED'}")
            lines.append("")
        lines += [
            "| id | char | orig | seed1 | seed2 | mean | std | min | max |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['id']} | {r['char']} | {r['orig'] if r['orig'] is not None else '-'} | "
                f"{r['a']} | {r['b']} | {r['mean']:.3f} | {r['std']:.3f} | {r['min']} | {r['max']} |"
            )
        lines.append("")
        return lines

    q14 = groups["q14_rag"]
    gold = groups["gold"]
    reachable = bool(gold) and (st.mean([r["mean"] for r in gold]) >= 4.3 or sum(1 for r in gold if r["mean"] >= 4.3) >= max(1, len(gold)//2))
    stable = True
    all_std = [r["std"] for r in per_item]
    mean_std = st.mean(all_std) if all_std else 0.0
    if mean_std > 0.4:
        stable = False
    lines = [
        "# exp6 judge sanity",
        "",
        "Judge: qwen3:14b through ollama, temp 0.0, JSON format, think off. Same prompt as exp4/judge.py (via voicefilt.py).",
        "Pass A: seed 1 (harness seed). Pass B: seed 2.",
        "q14_rag answers are re-scored only; exp4/runs files are not rewritten. Gold passages are on-page ch1-3 dialogue.",
        "",
        f"- per-item mean std (A vs B) across all items: {mean_std:.4f}",
        f"- 4.3 reachable on gold: {'yes' if reachable else 'no'}",
        f"- variance high (>0.4 mean std): {'yes — keep the same harness anyway' if not stable else 'no'}",
        "",
        "The harness is not changed. High variance is reported and the rest of exp6 continues with qwen3:14b temp 0 seed 1.",
        "",
    ]
    lines += block("q14_rag", q14)
    lines += block("gold", gold)
    # original q14 voice vs rejudge seed 1
    both = [r for r in q14 if r["orig"] is not None and r["a"] is not None]
    if both:
        diff = [r["a"] - r["orig"] for r in both]
        lines += [
            "## Original q14_rag score vs seed-1 rejudge",
            "",
            f"- n: {len(both)}",
            f"- mean(rejudge - orig): {st.mean(diff):.4f}",
            f"- |diff| mean: {st.mean(abs(x) for x in diff):.4f}",
            f"- orig mean: {st.mean(r['orig'] for r in both):.4f}",
            f"- seed-1 mean: {st.mean(r['a'] for r in both):.4f}",
            "",
        ]
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    open(md_path, "w").write(text)
    summary = dict(
        n_q14=len(q14), n_gold=len(gold), mean_std=mean_std, reachable=reachable, stable=stable,
        gold_ge_43=sum(1 for r in gold if r["mean"] >= 4.3),
        gold_n=len(gold),
        gold_mean=st.mean([r["mean"] for r in gold]) if gold else None,
        q14_mean=st.mean([r["mean"] for r in q14]) if q14 else None,
    )
    json.dump(dict(summary=summary, items=per_item), open(json_path, "w"), indent=1)
    print(json.dumps(summary), flush=True)
    print("wrote", md_path, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(H, "data"))
    ap.add_argument("--md", default=os.path.join(H, "judge_sanity.md"))
    ap.add_argument("--limit", type=int, default=0, help="cap q14 items (smoke)")
    ap.add_argument("--gold-n", type=int, default=8, help="gold lines per character")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    raw_path = os.path.join(a.out_dir, "judge_sanity.jsonl")
    items = q14_items(a.limit) + gold_items(a.gold_n if not a.limit else min(a.gold_n, 2))
    print("items", len(items), "q14", sum(1 for x in items if x["group"]=="q14_rag"),
          "gold", sum(1 for x in items if x["group"]=="gold"), flush=True)
    gpuutil.unload_ollama()
    run_pass(items, raw_path, seed=1, tag="A")
    run_pass(items, raw_path, seed=2, tag="B")
    summarize(load_jsonl(raw_path), items, a.md, os.path.join(a.out_dir, "judge_sanity_report.json"))

if __name__ == "__main__":
    main()
