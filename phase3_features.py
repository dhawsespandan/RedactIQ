"""Phase 3 (Assignment 3) - Feature engineering and word vectors.

Every token of a synthetic corpus becomes a context window of 3 words on each
side, with the token itself masked as <TARGET>, labelled 1 if it is PII.
BoW / TF-IDF features train Naive Bayes and Logistic Regression; the Logistic
Regression model is the CONTEXT CHECK of the pipeline, replacing A2's
hand-written trigger words (+0.05 when the context looks like PII).
word2vec CBOW and skip-gram are trained too; the skip-gram vectors (size 64)
initialise the embedding layer of the phase 4 BiLSTM.
"""
import random

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB

from phase1_preprocessing import token_sentences, tokenize

# ---------------- Synthetic corpus with exact PII spans (also used by phase 4) ----------------
NAMES = ["Rohan Mehta", "Anita Desai", "Kabir Rao", "Farhan Qureshi",
         "Priya Sharma", "Neha Kulkarni", "Arjun Patel", "Sara Khan"]
ORGS = ["Infosys Limited", "Wipro Technologies", "Symbiosis Institute"]


def pan():
    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5)) + \
        "".join(random.choices("0123456789", k=4)) + random.choice("ABCDEFGH")


def phone():
    return random.choice(["", "+91-"]) + str(random.randint(6, 9)) + \
        "".join(random.choices("0123456789", k=9))


def mail(n):
    return n.lower().replace(" ", ".") + "@sitpune.edu.in"


def build(parts):
    """parts = [(chunk, label or 0), ...] -> sentence text and its labelled PII spans."""
    text, spans = "", []
    for chunk, lab in parts:
        if lab:
            spans.append((len(text), len(text) + len(chunk), lab))
        text += chunk
    return text, spans


TEMPLATES = [
    lambda n: build([("Please contact ", 0), (n, "PERSON"), (" at ", 0), (mail(n), "EMAIL"),
                     (" or on ", 0), (phone(), "PHONE"), (" today.", 0)]),
    lambda n: build([("Employee ", 0), (n, "PERSON"), (" of ", 0), (random.choice(ORGS), "ORG"),
                     (" has PAN ", 0), (pan(), "PAN"), (".", 0)]),
    lambda n: build([("Ticket raised by ", 0), (n, "PERSON"), (". Registered mobile is ", 0),
                     (phone(), "PHONE"), (".", 0)]),
    lambda n: build([("Kindly forward the offer letter to ", 0), (n, "PERSON"), (" at ", 0),
                     (mail(n), "EMAIL"), (" before Friday.", 0)]),
    lambda n: build([("The applicant ", 0), (n, "PERSON"), (" submitted PAN ", 0),
                     (pan(), "PAN"), (" for verification.", 0)]),
    lambda n: build([("Our records show that ", 0), (n, "PERSON"), (" works with ", 0),
                     (random.choice(ORGS), "ORG"), (".", 0)]),
    lambda n: build([("The compliance review meeting is scheduled next week.", 0)]),
    lambda n: build([("All departments must submit the quarterly audit report.", 0)]),
    lambda n: build([("No personal identifiers are present in this document.", 0)]),
]
WIN = 3


def window(toks, i):
    """Three words on each side of token i, the token itself masked as <TARGET>."""
    left = [t[0].lower() for t in toks[max(0, i - WIN):i]]
    right = [t[0].lower() for t in toks[i + 1:i + 1 + WIN]]
    return " ".join(left + ["<TARGET>"] + right)


def windows(text, spans):
    """One (window, label) row per word token; label 1 if the token lies inside a PII span."""
    return [(window(toks, i), int(any(a <= s and e <= b for a, b, _ in spans)))
            for toks in token_sentences(text)
            for i, (w, s, e) in enumerate(toks) if any(c.isalnum() for c in w)]


def train(seed=42):
    """Build the corpus, train the four classifiers and both word2vec models; return everything."""
    random.seed(seed)
    corpus = [random.choice(TEMPLATES)(random.choice(NAMES)) for _ in range(800)]
    data = [r for t, s in corpus for r in windows(t, s)]
    random.Random(seed).shuffle(data)                  # shuffle before splitting 3 : 1
    cut = int(0.75 * len(data))
    Xtr, ytr = zip(*data[:cut])
    Xte, yte = zip(*data[cut:])

    bow = CountVectorizer(token_pattern=r"\S+", min_df=2)
    tfidf = TfidfVectorizer(token_pattern=r"\S+", min_df=2)
    Btr, Bte = bow.fit_transform(Xtr), bow.transform(Xte)
    Ttr, Tte = tfidf.fit_transform(Xtr), tfidf.transform(Xte)
    idf = sorted(zip(tfidf.get_feature_names_out(), tfidf.idf_), key=lambda x: x[1])

    results, models = [], {}
    for name, A, B, clf in [("BoW + MultinomialNB", Btr, Bte, MultinomialNB()),
                            ("TF-IDF + MultinomialNB", Ttr, Tte, MultinomialNB()),
                            ("BoW + LogisticReg", Btr, Bte, LogisticRegression(max_iter=1000)),
                            ("TF-IDF + LogisticReg", Ttr, Tte, LogisticRegression(max_iter=1000))]:
        pred = clf.fit(A, ytr).predict(B)
        P, R, F, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
        results.append({"Features + model": name, "Accuracy": round(accuracy_score(yte, pred), 3),
                        "Prec": round(P, 3), "Recall": round(R, 3), "F1": round(F, 3)})
        models[name] = clf

    nb = models["BoW + MultinomialNB"]                 # context words most indicative of PII
    ratio = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    top = [t for t, _ in sorted(zip(bow.get_feature_names_out(), ratio), key=lambda x: -x[1])
           if t.isalpha()][:8]

    # word2vec on the identifier-free vocabulary (tokens with digits or symbols are dropped)
    sents = [[w.lower() for w, _, _ in tokenize(t) if w.isalpha()] for t, _ in corpus]
    cbow = Word2Vec(sents, vector_size=64, window=5, min_count=2, sg=0,
                    epochs=40, seed=seed, workers=1)
    skip = Word2Vec(sents, vector_size=64, window=5, min_count=2, sg=1,
                    negative=10, epochs=40, seed=seed, workers=1)
    neighbours = {w: [(a, round(float(b), 2)) for a, b in skip.wv.most_similar(w, topn=4)]
                  for w in ["contact", "mobile", "pan"] if w in skip.wv}
    sims = [{"pair": f"{a} / {b}", "CBOW": round(float(cbow.wv.similarity(a, b)), 3),
             "skip-gram": round(float(skip.wv.similarity(a, b)), 3)}
            for a, b in [("contact", "mobile"), ("contact", "pan")] if a in skip.wv and b in skip.wv]

    return {"n_windows": len(data), "n_pii": sum(y for _, y in data), "bow_shape": Btr.shape,
            "idf_low": [t for t, _ in idf[:5]], "idf_high": [t for t, _ in idf[-3:]],
            "results": results, "top_words": top, "vec": bow, "clf": models["BoW + LogisticReg"],
            "w2v_vocab": len(skip.wv), "neighbours": neighbours, "sims": sims, "skip": skip.wv}


def context_prob(A3, text, e):
    """P(PII | context) for an entity: mean classifier probability over the tokens of its span."""
    rows = [window(toks, i) for toks in token_sentences(text)
            for i, (_, s, t) in enumerate(toks) if s < e["end"] and t > e["start"]]
    if not rows:
        return 0.0
    return float(A3["clf"].predict_proba(A3["vec"].transform(rows))[:, 1].mean())
