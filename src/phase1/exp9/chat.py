"""Chat-template ids. Qwen3 and SmolLM3 take enable_thinking; Phi does not."""
_mode = {}

def render(tok, messages, add_generation_prompt):
    key = id(tok)
    mode = _mode.get(key, "try")
    kw = dict(tokenize=True, return_dict=True, add_generation_prompt=add_generation_prompt)
    if mode == "think":
        out = tok.apply_chat_template(messages, enable_thinking=False, **kw)
    elif mode == "plain":
        out = tok.apply_chat_template(messages, **kw)
    else:
        try:
            out = tok.apply_chat_template(messages, enable_thinking=False, **kw)
            _mode[key] = "think"
        except TypeError:
            out = tok.apply_chat_template(messages, **kw)
            _mode[key] = "plain"
    ids = out["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)
