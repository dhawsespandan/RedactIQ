# RedactIQ - Explainable PII detection and redaction with confidence-scored audit trails.
# One Streamlit app that runs Assignments 1-6 as a single pipeline.
#
#   pip install -r requirements.txt      (includes spaCy's en_core_web_sm model)
#   streamlit run app.py
#
# If spaCy cannot be loaded the app still runs, with the spaCy detector switched off.
import html
import re
from collections import Counter

import pandas as pd
import streamlit as st

import phase1_preprocessing as p1
import phase2_detection as p2
import phase3_features as p3
import phase4_bilstm as p4
import phase5_translation as p5
import phase6_summary as p6

SAMPLE = """Subject: Verification of new joiners
Please contact Meera Joshi at meera.joshi@sitpune.edu.in or on +91-9871234560.
Employee Vikram Nair of Tata Motors has PAN BQWPK4821L.
Ticket raised by Ayesha Siddiqui. Registered mobile is 9812345670.
The applicant Dev Malhotra submitted PAN KJHGF5678D for verification.
Aadhaar 4321 8765 2109 belongs to Dev Malhotra.
Kindly forward the letter to the Placement Cell before Friday.
The compliance review meeting is scheduled next week.
The documents were uploaded on 14/10/2026.
Copy to: Registrar Office."""
COLORS = {"AUTO_REDACT": "#f5b7b1", "REVIEW": "#f9e79f", "LOG_ONLY": "#d5d8dc"}
TAG = r"\[[A-Z]+_REDACTED\]"

st.set_page_config(page_title="RedactIQ", page_icon="🛡️", layout="wide")


# ---------------- Models: trained once at startup on their built-in synthetic data ----------------
@st.cache_resource(show_spinner="A2 · loading spaCy en_core_web_sm ...")
def load_spacy():
    return p2.load_spacy()


@st.cache_resource(show_spinner="A3 · training context classifiers and word2vec ...")
def train_a3():
    return p3.train()


@st.cache_resource(show_spinner="A4 · training the BiLSTM tagger and its ablation ...")
def train_a4():
    return p4.train(train_a3()["skip"])


@st.cache_resource(show_spinner="A5 · training the seq2seq translator ...")
def train_a5():
    return p5.train()


@st.cache_resource(show_spinner="A6 · pre-training the mini-BERT ...")
def train_a6():
    return p6.train()


@st.cache_resource(show_spinner="A2 · evaluating detector configurations ...")
def evaluate_a2():
    rx, sp, bl = p2.regex_detect, lambda t: p2.spacy_detect(t, nlp), lambda t: p4.detect(a4, t)
    return [{"Configuration": name, **p2.evaluate(d)} for name, d in [
        ("C1 regex only", [rx]), ("C2 spaCy NER only", [sp]),
        ("C3 fusion of both", [rx, sp]), ("C4 fusion + BiLSTM", [rx, sp, bl])]]


nlp, a3, a4, a5, a6 = load_spacy(), train_a3(), train_a4(), train_a5(), train_a6()
a2_eval = evaluate_a2()


def box(inner):
    return (f'<div style="white-space:pre-wrap;line-height:1.7;padding:12px 14px;border-radius:8px;'
            f'border:1px solid rgba(128,128,128,.35)">{inner}</div>')


def marked(text, ents):
    """Original text with every entity highlighted by its decision band."""
    out, pos = [], 0
    for e in sorted(ents, key=lambda x: x["start"]):
        if e["start"] < pos:
            continue
        tip = f"{e['label']} · conf {e['conf']:.2f} · {e['band']}"
        out += [html.escape(text[pos:e["start"]]),
                f'<mark title="{tip}" style="background:{COLORS[e["band"]]};color:#000;border-radius:3px">'
                f'{html.escape(text[e["start"]:e["end"]])}</mark>']
        pos = e["end"]
    return box("".join(out) + html.escape(text[pos:]))


def tags(text):
    """Redacted text with the placeholders highlighted."""
    t = re.sub(TAG, lambda m: f'<mark style="background:#1f2d3d;color:#fff;border-radius:3px">{m.group()}</mark>',
               html.escape(text))
    return box(t.replace("[not translated]", '<span style="opacity:.55">[not translated]</span>'))


def table(rows):
    st.dataframe(pd.DataFrame(rows), hide_index=True)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    st.caption("Detectors")
    use_rx = st.toggle("Regex rules (A2)", True)
    use_sp = st.toggle("spaCy NER (A2)", nlp is not None, disabled=nlp is None)
    if nlp is None:
        st.warning("spaCy could not be loaded, so the spaCy detector is off.")
    use_bl = st.toggle("BiLSTM tagger (A4)", True)
    ctx_mode = st.radio("Context boost (+0.05)", ["A3 learned classifier", "A2 trigger words", "Off"])
    st.caption("Decision thresholds")
    auto = st.slider("AUTO_REDACT at or above", 0.50, 0.99, p2.AUTO, 0.01)
    review = min(st.slider("REVIEW at or above", 0.05, 0.95, p2.REVIEW, 0.01), auto)
    st.caption(f"LOG_ONLY below {review:.2f} (logged, not redacted)")

st.title("🛡️ RedactIQ")
st.caption("Explainable PII detection and redaction with confidence-scored audit trails · Assignments 1-6 in one pipeline")

# ---------------- 1 · Input and tidy (A1) ----------------
st.subheader("1 · Document")
c1, c2 = st.columns([3, 1])
up = c1.file_uploader("Upload one .txt document", type=["txt"])
use_sample = c2.toggle("Use sample document")
if up is not None:
    raw, source = up.getvalue().decode("utf-8", errors="replace"), up.name
elif use_sample:
    raw, source = SAMPLE, "sample document"
else:
    st.info("Upload a .txt file or switch on the sample document.")
    st.stop()
doc = p1.tidy(raw)
if not doc:
    st.warning("The document is empty.")
    st.stop()
pre = p1.preprocess(doc)
st.caption(f"Using **{source}** · tidied to {len(doc)} characters, {len(pre['sentences'])} sentences, "
           f"{len(pre['tokens'])} tokens (case and PII kept intact).")

# ---------------- 2 · Three detectors -> context check -> merge, score, decide ----------------
st.subheader("2 · Detect, check context, score and decide")
dets = {"regex": p2.regex_detect(doc) if use_rx else [],
        "spaCy": p2.spacy_detect(doc, nlp) if use_sp else [],
        "BiLSTM": p4.detect(a4, doc) if use_bl else []}
context = {"A3 learned classifier": lambda t, e: p3.context_prob(a3, t, e),
           "A2 trigger words": p2.trigger_rule, "Off": lambda t, e: 0.0}[ctx_mode]
ents = p2.score(p2.fuse([d for v in dets.values() for d in v]), doc, context, auto, review)
n_fused = len(ents)
bands = Counter(e["band"] for e in ents)
m = st.columns(6)
m[0].metric("Regex", len(dets["regex"]))
m[1].metric("spaCy NER", len(dets["spaCy"]))
m[2].metric("BiLSTM", len(dets["BiLSTM"]))
m[3].metric("AUTO_REDACT", bands["AUTO_REDACT"])
m[4].metric("REVIEW", bands["REVIEW"])
m[5].metric("LOG_ONLY", bands["LOG_ONLY"])
st.caption(f"{sum(map(len, dets.values()))} raw detections fused into {len(ents)} entities · "
           f"context boost: {ctx_mode} · bands: >= {auto:.2f} AUTO_REDACT, >= {review:.2f} REVIEW")

# ---------------- 3 · Human review of the medium-confidence items ----------------
review_items = [e for e in ents if e["band"] == "REVIEW"]
if review_items:
    st.subheader("3 · Review checklist")
    st.caption("These scored between the two thresholds. They are redacted unless you untick them; "
               "your choice applies to every occurrence of the value.")
    for e in review_items:
        e["redact"] = st.checkbox(
            f"Redact `{e['text']}` · {e['label']} · conf {e['conf']:.2f} · votes {'+'.join(dict.fromkeys(e['src']))}",
            value=True, key=f"review-{e['start']}-{e['end']}-{e['text']}")
ents = p2.propagate(doc, ents)                     # a value redacted once is redacted everywhere
repeats = sum("repeat_of" in e for e in ents)
redacted = p2.redact(doc, ents)

# ---------------- 4 · Original vs redacted ----------------
st.subheader("4 · Original vs redacted")
left, right = st.columns(2)
with left:
    st.markdown("**Original** · 🟥 auto-redact · 🟨 review · ⬜ log only")
    st.html(marked(doc, ents))
with right:
    st.markdown("**Redacted**")
    st.html(tags(redacted))
with st.expander("Explainable audit trail", expanded=True):
    if ents:
        table(p2.audit_rows(ents))
    else:
        st.write("No PII was detected.")

# ---------------- 5 · Hindi translation of the redacted document (A5) ----------------
st.subheader("5 · Hindi translation (romanised)")
hindi, tr_rows = p5.translate_document(a5, redacted)
n_ok = sum(bool(r["Model input"]) for r in tr_rows)
leaked = p5.leaks(hindi, [e["text"] for e in ents if e["redact"]])
st.html(tags(hindi))
st.caption(f"{n_ok} of {len(tr_rows)} sentences translated, {len(tr_rows) - n_ok} marked not translated · "
           f"redacted values found in the translation: {len(leaked)} "
           f"{'✅' if not leaked else '❌ ' + ', '.join(leaked)}")

# ---------------- 6 · Audit summary (A6) ----------------
st.subheader("6 · Audit summary")
audit_sents = [p6.describe(e) for e in ents]
picked, clusters = p6.summarise(a6["model"], audit_sents)
if audit_sents:
    for i in picked:
        st.markdown(f"- [{i + 1}] {audit_sents[i]}")
    st.caption(f"{len(audit_sents)} audit sentences -> {len(picked)} "
               f"({100 * len(picked) / len(audit_sents):.0f}% retained), original order kept.")
else:
    st.write("No PII was detected, so there is nothing to summarise.")

# ---------------- 7 · Download ----------------
st.subheader("7 · Download")
masked_rows = [{**r, "Entity": "***" if r["Action"].startswith("redacted") else r["Entity"]}
               for r in p2.audit_rows(ents)]
report = "\n\n".join([
    "REDACTIQ REPORT", "REDACTED DOCUMENT\n" + redacted, "HINDI TRANSLATION (romanised)\n" + hindi,
    "AUDIT SUMMARY\n" + "\n".join(f"[{i + 1}] {audit_sents[i]}" for i in picked),
    "AUDIT TRAIL (redacted values masked)\n" + (pd.DataFrame(masked_rows).to_string(index=False) if ents else "none")])
d1, d2 = st.columns(2)
d1.download_button("⬇ Redacted document (.txt)", redacted, "redacted.txt", "text/plain")
d2.download_button("⬇ Full report (.txt)", report, "redactiq_report.txt", "text/plain")

# ---------------- What each phase did in this run ----------------
st.divider()
st.subheader("What each phase did in this run")

with st.container(border=True):
    found = Counter(p[0] for p in pre["pii"])
    st.markdown(
        f"**A1 · Text preprocessing (NLTK)** · {len(pre['sentences'])} sentences, {len(pre['tokens'])} word tokens. "
        f"Grabbed {len(pre['pii'])} formatted PII items **before** cleaning "
        f"({', '.join(f'{k} {v}' for k, v in found.items()) or 'none'}); after cleaning {pre['survived']} of them "
        f"survive, so detection runs on the tidied original. Stop words: {len(pre['words'])} -> {len(pre['filtered'])} tokens; "
        f"stem and lemma differ for {len(pre['differ'])} words.")
    with st.expander("A1 details"):
        st.write("Sentences:", pre["sentences"])
        table([{"Label": l, "Span": f"{s}-{e}", "Text": t} for l, s, e, t in pre["pii"]] or [{"Label": "none"}])
        st.write("Cleaned text:", pre["cleaned"])
        table([{"Token": w, "Porter stem": s, "WordNet lemma": l} for w, s, l in pre["norm"]])

with st.container(border=True):
    merged = sum(map(len, dets.values())) - n_fused
    st.markdown(
        f"**A2 · Detection, fusion and scoring** · regex {len(dets['regex'])} + spaCy {len(dets['spaCy'])} "
        f"(+ BiLSTM {len(dets['BiLSTM'])}) detections -> {n_fused} entities ({merged} duplicates merged by "
        f"IoU / containment). Noisy-OR confidence gave {bands['AUTO_REDACT']} AUTO_REDACT, {bands['REVIEW']} REVIEW, "
        f"{bands['LOG_ONLY']} LOG_ONLY; {sum(e['redact'] for e in ents)} redacted right to left, "
        f"{repeats} of them repeats of a value already redacted elsewhere.")
    with st.expander("A2 details"):
        if ents:
            e = max(ents, key=lambda x: len(x["sc"]))
            prod = " x ".join(f"(1 - {s:.2f})" for s in e["sc"])
            st.write(f"Example: `{e['text']}` has votes {e['sc']} -> conf = 1 - {prod}"
                     f"{' + 0.05 context' if e['ctx'] >= 0.5 else ''} = {e['conf']:.3f}"
                     f"{' (capped at 0.99)' if e['conf'] == 0.99 else ''} -> {e['band']}")
        st.write("Evaluation on the labelled test set of Assignment 2:")
        table(a2_eval)

with st.container(border=True):
    lr, nb = a3["results"][2], a3["results"][0]
    boosted = sum(e["ctx"] >= 0.5 for e in ents)
    st.markdown(
        f"**A3 · Context features** · Logistic Regression on Bag-of-Words context windows (3 words each side, entity "
        f"masked as `<TARGET>`) reached F1 {lr['F1']} vs Naive Bayes {nb['F1']} on {a3['n_windows']} windows. "
        f"It is the context check: here {boosted} of {len(ents)} entities got the +0.05 boost ({ctx_mode}). "
        f"Skip-gram vectors ({a3['w2v_vocab']} words, 64-d) initialised the BiLSTM.")
    with st.expander("A3 details"):
        st.write(f"BoW matrix {a3['bow_shape']} · lowest IDF {a3['idf_low']} · highest IDF {a3['idf_high']}")
        table(a3["results"])
        st.write("Context words most indicative of PII (Naive Bayes):", ", ".join(a3["top_words"]))
        st.write("Skip-gram nearest neighbours:", a3["neighbours"])
        table(a3["sims"])

with st.container(border=True):
    only_bl = [e for e in ents if set(e["src"]) == {"bilstm"} and "repeat_of" not in e]
    caught = [e["text"] for e in only_bl if e["band"] != "LOG_ONLY"]
    guesses = [e["text"] for e in only_bl if e["band"] == "LOG_ONLY"]
    st.markdown(
        f"**A4 · BiLSTM tagger (third detector)** · found {len(dets['BiLSTM'])} entities here, "
        f"{sum(e['unseen'] for e in dets['BiLSTM'])} of them made only of words it never saw in training "
        f"(recognised from shape and context). Caught with no regex or spaCy vote: {', '.join(caught) or 'none'}; "
        f"low-confidence guesses, only logged: {', '.join(guesses) or 'none'}. "
        f"Entity F1 on unseen test names {a4['report'][-1]['F1']}; ablation "
        f"{' -> '.join(str(r['F1']) for r in a4['ablation'])} (embedding only -> + shape -> + word dropout).")
    with st.expander("A4 details"):
        st.write(f"{a4['n_train']} training / {a4['n_test']} test sentences · vocabulary {len(a4['vocab'])} words "
                 f"({a4['w2v_init']} initialised from skip-gram) · {a4['oov_test']} test tokens out of vocabulary · "
                 f"{a4['params']:,} parameters · loss {a4['losses'][0]:.4f} -> {a4['losses'][-1]:.4f}")
        st.write("BIO example:", a4["bio_example"])
        table(a4["report"])
        table(a4["ablation"])

with st.container(border=True):
    (lex_src, _), (dlx_src, _) = a5["sizes"]["lexicalised"], a5["sizes"]["delexicalised"]
    st.markdown(
        f"**A5 · Encoder-decoder translation** · {n_ok} of {len(tr_rows)} sentences translated to romanised Hindi "
        f"with Bahdanau attention; redaction tags bypass the network and are restored, {len(leaked)} PII leaks. "
        f"Delexicalised source vocabulary: {dlx_src} words (vs {lex_src} if identifiers were left in).")
    with st.expander("A5 details"):
        st.write(f"{a5['params']:,} parameters · loss {a5['losses'][0]:.4f} -> {a5['losses'][-1]:.4f} · "
                 f"training pair: {a5['pairs'][1]}")
        table(tr_rows)

with st.container(border=True):
    st.markdown(
        f"**A6 · mini-BERT audit summary** · {len(audit_sents)} audit sentences -> {len(picked)} by k-means over "
        f"mean-pooled embeddings. MLM loss {a6['losses'][0]:.3f} -> {a6['losses'][-1]:.3f}; the word 'name' in two "
        f"sentences has cosine {a6['cosine']} (word2vec would give 1.000).")
    with st.expander("A6 details"):
        for s, top in a6["fills"]:
            st.write(f"`{s}` -> {', '.join(top)}")
        table([{"#": i + 1, "Audit sentence": s, "Cluster": c + 1, "In summary": i in picked}
               for i, (s, c) in enumerate(zip(audit_sents, clusters))] or [{"#": "none"}])
