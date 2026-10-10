"""Phase 2 (Assignment 2) - PII detection, fusion, confidence scoring and redaction.

Detector 1 = regular expressions, Detector 2 = spaCy NER (Detector 3, the BiLSTM, is phase 4).
Detections of the same entity are fused (IoU > 0.5 or containment), scored by a
noisy-OR over the detector votes plus a context boost, mapped to a decision band
(AUTO_REDACT / REVIEW / LOG_ONLY) and replaced right to left. Every decision is
kept as a row of the audit trail.
"""
import re

PATTERNS = {                                   # regex and the base score of each
    "EMAIL":   (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", 0.85),
    "PHONE":   (r"(?:\+91[-\s]?|0)?[6-9]\d{9}\b", 0.85),
    "AADHAAR": (r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b", 0.95),
    "PAN":     (r"\b[A-Z]{5}\d{4}[A-Z]\b", 0.95),
}
SPACY_MAP = {"PERSON": "PERSON", "ORG": "ORG"}
BLOCK = {"pan", "aadhaar", "dob"}              # field names, not values
TRIGGERS = ("email", "contact", "phone", "mobile", "pan", "aadhaar")
AUTO, REVIEW = 0.85, 0.50                      # default decision thresholds


def load_spacy():
    """en_core_web_sm, or None if spaCy cannot be imported or loaded (the detector is then skipped)."""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def regex_detect(text):
    """Detector 1 : rule based."""
    return [{"text": m.group(), "start": m.start(), "end": m.end(),
             "label": lab, "src": ["regex"], "sc": [base]}
            for lab, (pat, base) in PATTERNS.items()
            for m in re.finditer(pat, text)]


def spacy_detect(text, nlp):
    """Detector 2 : statistical NER."""
    if nlp is None:
        return []
    return [{"text": e.text, "start": e.start_char, "end": e.end_char,
             "label": SPACY_MAP[e.label_], "src": ["spacy"], "sc": [0.75]}
            for e in nlp(text).ents
            if e.label_ in SPACY_MAP and e.text.lower() not in BLOCK]


def iou(a, b):
    """Intersection over union of two character spans."""
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union else 0.0


def same(a, b):
    """Same entity if the spans overlap well or one contains the other."""
    return (iou(a, b) > 0.5
            or (a["start"] <= b["start"] and b["end"] <= a["end"])
            or (b["start"] <= a["start"] and a["end"] <= b["end"]))


def fuse(ents):
    """Merge duplicate detections, keeping the longer span and every vote."""
    merged = []
    for e in sorted(ents, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
        for m in merged:
            if same(m, e):
                if e["end"] - e["start"] > m["end"] - m["start"]:
                    m.update(text=e["text"], start=e["start"], end=e["end"])
                m["src"] = m["src"] + e["src"]
                m["sc"] = m["sc"] + e["sc"]
                break
        else:
            merged.append(dict(e))
    return merged


def trigger_rule(text, e):
    """A2's hand-written context rule (phase 3 replaces it with a learned classifier)."""
    return float(any(w in text[max(0, e["start"] - 30):e["start"]].lower() for w in TRIGGERS))


def confidence(e, boost):
    """Noisy-OR over the detector votes, plus 0.05 when the context looks like PII."""
    p = 1.0
    for s in e["sc"]:
        p *= 1 - s
    return min(1 - p + (0.05 if boost else 0.0), 0.99)


def band(c, auto=AUTO, review=REVIEW):
    return "AUTO_REDACT" if c >= auto else ("REVIEW" if c >= review else "LOG_ONLY")


def score(ents, text, context=trigger_rule, auto=AUTO, review=REVIEW):
    """Context check, confidence and decision for every fused entity."""
    for e in ents:
        e["ctx"] = context(text, e)
        e["conf"] = confidence(e, e["ctx"] >= 0.5)
        e["band"] = band(e["conf"], auto, review)
        e["redact"] = e["band"] != "LOG_ONLY"       # REVIEW items may be un-ticked by a human
    return ents


def redact(text, ents):
    """Replace every entity marked for redaction, right to left so earlier offsets stay valid."""
    last = len(text)
    for e in sorted(ents, key=lambda x: -x["start"]):
        if e["redact"]:
            text = text[:e["start"]] + f"[{e['label']}_REDACTED]" + text[min(e["end"], last):]
            last = e["start"]
    return text


def audit_rows(ents):
    """The explainable audit trail: span, type, votes, context, confidence and decision."""
    return [{"Entity": e["text"], "Span": f"{e['start']}-{e['end']}", "Type": e["label"],
             "Votes": "+".join(dict.fromkeys(e["src"])),
             "Vote scores": ", ".join(f"{s:.2f}" for s in e["sc"]),
             "Context": round(e["ctx"], 2), "Conf": round(e["conf"], 3), "Decision": e["band"],
             "Action": "redacted" if e["redact"] else "kept"} for e in ents]


# ---------------- Evaluation on a small labelled test set ----------------
GOLD = [
    ("Employee Kabir Rao of Wipro Technologies has PAN ZXCVB9876K.",
     [("Kabir Rao", "PERSON"), ("Wipro Technologies", "ORG"), ("ZXCVB9876K", "PAN")]),
    ("Mail Anita Desai at anita.desai@gmail.com or call 9812345670.",
     [("Anita Desai", "PERSON"), ("anita.desai@gmail.com", "EMAIL"), ("9812345670", "PHONE")]),
    ("Aadhaar 5678 1234 9012 belongs to Farhan Qureshi.",
     [("5678 1234 9012", "AADHAAR"), ("Farhan Qureshi", "PERSON")]),
    ("The compliance review meeting is scheduled next week.", []),
]


def evaluate(detectors):
    """Span-level TP / FP / FN, precision, recall, F1 and leakage for one set of detectors."""
    tp = fp = fn = 0
    for text, gold in GOLD:
        pred = [(e["text"], e["label"]) for e in fuse([d for f in detectors for d in f(text)])]
        left = list(gold)
        for p in pred:
            if p in left:
                left.remove(p)
                tp += 1
            else:
                fp += 1
        fn += len(left)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "Prec": round(P, 3), "Recall": round(R, 3),
            "F1": round(F, 3), "Leakage": round(1 - R, 3)}
