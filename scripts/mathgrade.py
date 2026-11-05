"""Answer extraction and grading, shared by the eval scripts and the agent harness.

Extraction follows the authors' protocol (exp8_math/code/mathqa.py): if the response contains '</think>', only the text
after the last '</think>' is read; otherwise the whole response. The answer is the content of the LAST \\boxed{...}
(brace matched). No closed box = no answer = wrong.
Check: math-verify, verify(parse('\\boxed{gold}'), parse('\\boxed{pred}')); a plain normalized string match is
reported separately as `correct_str` for reference.
"""
import re

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def build_user_message(problem):
    return problem + "\n" + INSTRUCTION


def answer_region(text):
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i = text.find("{", idx)
    if i < 0 or i > idx + 8:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def _norm(s):
    s = s.strip().replace(" ", "").replace("\\!", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s).rstrip(".").replace("$", "")
    return s


_mv = None


def grade(response, gold):
    """Returns dict(pred, correct, correct_str)."""
    global _mv
    pred = last_boxed(answer_region(response))
    if pred is None:
        return {"pred": None, "correct": False, "correct_str": False}
    if _mv is None:
        from math_verify import parse, verify
        _mv = (parse, verify)
    parse, verify = _mv
    try:
        ok = bool(verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + pred + "}")))
    except Exception:
        ok = False
    return {"pred": pred, "correct": ok, "correct_str": _norm(pred) == _norm(gold)}
