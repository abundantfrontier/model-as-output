"""Pass-2 rewrite prompt and the fact-diff judge prompt.

Pass 2 uses the character card (common.sysprompt) or the narrator card.
It does not call common.retrieve and does not use common.rag_sysprompt.
"""
import common as C

NARR = ("You are the narrator of the story No Return. Answer the reader's question about what has happened so far "
        "in plain, spare prose, 1-4 sentences. Never reveal or hint at what has not happened yet.")

REWRITE_INSTR = (
    "Rewrite the answer above in your voice. Change style only. "
    "Do not add, drop, or change any facts. Do not mention that this is a rewrite."
)

RAG_MARK = "Story so far (excerpts):"

# Stable text hashed into factdiff outputs. The question and the two answers are appended after it.
FACT_DIFF_HEAD = (
    "You are a strict fact checker. PASS1 is the source answer. PASS2 is a rewrite that was required to change style only and to keep every fact.\n"
    "Compare the concrete claims in the two answers. A claim is a yes/no, a number, a name, a place, an object, who did what, "
    "what a list or tag or dial says, a weight, a heading, a color, or a count.\n"
    "Rules:\n"
    "- A paraphrase that states the same proposition is not a drop and not an add.\n"
    "- Tone, word order, sentence length, and staying in character are not claims.\n"
    "- List a dropped claim when PASS1 states a proposition and PASS2 does not state that same proposition.\n"
    "- List an added claim when PASS2 states a proposition that PASS1 does not state. A new inference counts as an add. Restating the question does not.\n"
    "- Do not list claims that neither answer made.\n"
    "- Use an empty list when there is nothing in that category.\n"
    "Reply with JSON only, no markdown, in this shape: {\"dropped\": [], \"added\": []}\n"
    "Put each claim as a short string inside those arrays.\n"
)

def system_for(char, day=3):
    if char == "narrator":
        return NARR
    if char not in C.CARDS:
        raise KeyError(f"no card for {char}")
    return C.sysprompt(char, day)

def rewrite_user(question, answer):
    """Reader question, then the pass-1 answer, then the style-only instruction."""
    return question.rstrip() + "\n\n" + (answer or "").strip() + "\n\n" + REWRITE_INSTR

def fact_diff_prompt(question, pass1, pass2):
    return (
        FACT_DIFF_HEAD
        + "\nQUESTION:\n" + (question or "")
        + "\n\nPASS1:\n" + (pass1 or "")
        + "\n\nPASS2:\n" + (pass2 or "")
        + "\n"
    )

def assert_no_retrieval(system, user):
    """The rewrite prompt must not carry a retrieved passage block."""
    if RAG_MARK in system or RAG_MARK in user:
        raise RuntimeError("retrieval block leaked into the rewrite prompt")
