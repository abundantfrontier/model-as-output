"""DPO starts from the combo contrast adapter. Skip below 16 preference pairs."""
import os, json
import frozen

H = os.path.dirname(os.path.abspath(__file__))
MIN_PAIRS = 16

def main():
    os.makedirs(os.path.join(H, "data"), exist_ok=True)
    w = json.load(open(os.path.join(H, "data", "winner.json")))
    meta = {}
    mp = os.path.join(H, "data", "dpo_pairs_meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))
    pairs = meta.get("preference_pairs")
    if pairs is None:
        pairs = meta.get("pairs") or 0
        if os.path.exists(os.path.join(H, "data", "v_dpo_pairs.jsonl")):
            pairs = sum(1 for line in open(os.path.join(H, "data", "v_dpo_pairs.jsonl")) if line.strip() and json.loads(line).get("bad"))
    adapter = os.path.join(H, "lora_v_combo_contrast")
    if pairs < MIN_PAIRS or not os.path.isfile(os.path.join(adapter, "train_report.json")):
        why = []
        if pairs < MIN_PAIRS:
            why.append(f"only {pairs} preference pairs (need {MIN_PAIRS})")
        if not os.path.isfile(os.path.join(adapter, "train_report.json")):
            why.append("contrast adapter missing")
        doc = dict(skip=True, reason="; ".join(why), pairs=pairs, winner=w)
        json.dump(doc, open(os.path.join(H, "data", "dpo_choice.json"), "w"), indent=1)
        json.dump(doc, open(os.path.join(H, "data", "dpo_skipped.json"), "w"), indent=1)
        print(json.dumps(doc), flush=True)
        return
    doc = dict(
        skip=False, adapter=adapter, name="v_combo_contrast", pairs=pairs,
        mode=w["mode"], winner=w["winner"],
        reason="DPO starts from the contrast adapter as specified",
    )
    json.dump(doc, open(os.path.join(H, "data", "dpo_choice.json"), "w"), indent=1)
    print(json.dumps(doc), flush=True)

if __name__ == "__main__":
    main()
