"""Phase 4 (Assignment 4) - BiLSTM BIO tagger, the THIRD PII detector.

Word embedding (initialised from the phase 3 skip-gram vectors) + word-shape
embedding -> bidirectional LSTM -> one BIO tag per token. Word dropout hides
real words during training, so the model must use shape and context and
generalises to names it has never seen. Test names, organisations and
distractors are disjoint from the training ones.
In the pipeline each predicted entity votes 0.70 x (mean tag probability).
"""
import random

import torch
import torch.nn as nn

from phase1_preprocessing import token_sentences
from phase3_features import build, mail, pan, phone

NAMES_TR = ["Rohan Mehta", "Anita Desai", "Kabir Rao", "Farhan Qureshi", "Priya Sharma", "Neha Kulkarni"]
NAMES_TE = ["Meera Joshi", "Vikram Nair", "Ayesha Siddiqui", "Dev Malhotra"]
ORGS_TR = ["Infosys Limited", "Wipro Technologies", "Symbiosis Institute"]
ORGS_TE = ["Persistent Systems", "Tata Motors"]
DIST_TR = ["Registrar Office", "Human Resources", "Finance Department"]   # Title-case but NOT PII
DIST_TE = ["Placement Cell", "Academic Council", "Transport Section"]

TEMPLATES = [   # (n)ame, (o)rganisation and (d)istractor are substituted into each template
    lambda n, o, d: build([("Please contact ", 0), (n, "PERSON"), (" at ", 0), (mail(n), "EMAIL"),
                           (" or on ", 0), (phone(), "PHONE"), (".", 0)]),
    lambda n, o, d: build([("Employee ", 0), (n, "PERSON"), (" of ", 0), (o, "ORG"),
                           (" has PAN ", 0), (pan(), "PAN"), (".", 0)]),
    lambda n, o, d: build([("Ticket raised by ", 0), (n, "PERSON"),
                           (". Registered mobile is ", 0), (phone(), "PHONE"), (".", 0)]),
    lambda n, o, d: build([("The applicant ", 0), (n, "PERSON"), (" submitted PAN ", 0),
                           (pan(), "PAN"), (" for verification.", 0)]),
    lambda n, o, d: build([("Our records show that ", 0), (n, "PERSON"), (" works with ", 0),
                           (o, "ORG"), (".", 0)]),
    lambda n, o, d: build([("The compliance review meeting is scheduled next week.", 0)]),
    lambda n, o, d: build([("Kindly forward the letter to the ", 0), (d, 0), (" before Friday.", 0)]),
    lambda n, o, d: build([("Please contact the ", 0), (d, 0), (" team on Monday.", 0)]),
    lambda n, o, d: build([("The ", 0), (d, 0), (" will review the request in March.", 0)]),
]
LABELS = ["PERSON", "ORG", "EMAIL", "PHONE", "PAN"]
TAGS = ["O"] + [p + "-" + l for l in LABELS for p in ("B", "I")]
tag2i = {t: i for i, t in enumerate(TAGS)}
PAD, UNK, MAXLEN = 0, 1, 24


def to_bio(toks, spans):
    """B- marks the first token of an entity, I- the rest, O everything else."""
    words, tags = [], []
    for w, s, e in toks:
        tag = "O"
        for a, b, lab in spans:
            if a <= s and e <= b:
                tag = ("B-" if s == a else "I-") + lab
                break
        words.append(w)
        tags.append(tag)
    return words, tags


def shape(w):
    """8 word-shape classes, so an unseen name or identifier is recognised by its FORM."""
    d = w.lstrip("+").replace("-", "")
    if "@" in w:                                     return 1   # e-mail
    if d.isdigit():                                  return 2 if len(d) >= 10 else 3
    if w.isalpha() and w.istitle():                  return 4   # Titlecase word
    if w.isalpha() and w.isupper():                  return 5   # ALLCAPS
    if w.isalnum() and w.isupper() and any(c.isdigit() for c in w):
        return 6                                                # PAN-like AAAAA9999A
    if w.isalpha():                                  return 7   # lowercase word
    return 0


def encode(words, tags, vocab):
    x = [vocab.get(w.lower(), UNK) for w in words][:MAXLEN]
    s = [shape(w) for w in words][:MAXLEN]
    y = [tag2i[t] for t in tags][:MAXLEN]
    p = MAXLEN - len(x)
    return x + [PAD] * p, s + [0] * p, y + [-100] * p    # -100 is ignored by the loss


def tensors(rows, vocab):
    a, b, c = zip(*[encode(w, t, vocab) for w, t in rows])
    return torch.tensor(a), torch.tensor(b), torch.tensor(c)


class BiLSTMTagger(nn.Module):
    def __init__(self, nvocab, ntags, emb=64, shp=16, hid=96, use_shape=True):
        super().__init__()
        self.use_shape = use_shape
        self.emb = nn.Embedding(nvocab, emb, padding_idx=PAD)
        self.shape_emb = nn.Embedding(8, shp)
        self.lstm = nn.LSTM(emb + (shp if use_shape else 0), hid, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(hid * 2, ntags)          # hid*2 because it is bidirectional

    def forward(self, x, s):
        z = self.emb(x)
        if self.use_shape:
            z = torch.cat([z, self.shape_emb(s)], dim=-1)
        h, _ = self.lstm(z)
        return self.fc(self.drop(h))


def word_dropout(x, p=0.4):
    """Randomly replace real words by <unk> during training, so names are not memorised."""
    m = (torch.rand_like(x, dtype=torch.float) < p) & (x != PAD)
    return torch.where(m, torch.full_like(x, UNK), x)


def fit(D, use_shape=True, use_drop=True, epochs=25):
    """Train one tagger with Adam and a cross-entropy loss that ignores padding."""
    torch.manual_seed(42)
    m = BiLSTMTagger(len(D["vocab"]), len(TAGS), use_shape=use_shape)
    with torch.no_grad():                            # initialise from the skip-gram vectors
        for w, i in D["vocab"].items():
            if w in D["w2v"]:
                m.emb.weight[i] = torch.tensor(D["w2v"][w])
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    X, S, Y = D["train"]
    losses = []
    for _ in range(epochs):
        m.train()
        tot = 0.0
        for i in range(0, len(X), 32):
            xb, sb, yb = X[i:i + 32], S[i:i + 32], Y[i:i + 32]
            opt.zero_grad()
            loss = lossf(m(word_dropout(xb) if use_drop else xb, sb).reshape(-1, len(TAGS)), yb.reshape(-1))
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        losses.append(tot / len(X))
    return m.eval(), losses


def spans_from_bio(tags):
    """Decode a BIO tag sequence back into (start, end, label) token spans."""
    out, cur = [], None
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if cur: out.append(cur)
            cur = [i, i + 1, t[2:]]
        elif t.startswith("I-") and cur and cur[2] == t[2:]:
            cur[1] = i + 1
        else:
            if cur: out.append(cur)
            cur = None
    if cur: out.append(cur)
    return [tuple(s) for s in out]


def evaluate(m, D):
    """Entity-level precision / recall / F1 on the unseen test set, overall and per type."""
    X, S, Y = D["test"]
    with torch.no_grad():
        pred = m(X, S).argmax(-1)
    per = {l: [0, 0, 0] for l in LABELS}             # TP, FP, FN per entity type
    for r in range(len(X)):
        keep = Y[r] != -100
        g = spans_from_bio([TAGS[i] for i in Y[r][keep].tolist()])
        p_ = spans_from_bio([TAGS[i] for i in pred[r][keep].tolist()])
        for s in p_:
            per[s[2]][0 if s in g else 1] += 1
        for s in g:
            if s not in p_:
                per[s[2]][2] += 1
    per["OVERALL"] = [sum(v[k] for v in per.values()) for k in range(3)]
    rows = []
    for lab, (tp, fp, fn) in per.items():
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"Entity": lab, "TP": tp, "FP": fp, "FN": fn, "Prec": round(P, 3),
                     "Recall": round(R, 3), "F1": round(2 * P * R / (P + R) if P + R else 0.0, 3)})
    return rows


def make(n, names, orgs, dists):
    """n template sentences -> BIO rows, split into sentences exactly as detect() does."""
    raw = [random.choice(TEMPLATES)(random.choice(names), random.choice(orgs), random.choice(dists))
           for _ in range(n)]
    return [to_bio(toks, spans) for text, spans in raw for toks in token_sentences(text)]


def train(w2v, seed=42):
    """Corpus -> BIO -> vocabulary -> main model + ablation study; return everything."""
    random.seed(seed)
    train_bio = make(900, NAMES_TR, ORGS_TR, DIST_TR)
    test_bio = make(250, NAMES_TE, ORGS_TE, DIST_TE)
    counts = {}
    for ws, _ in train_bio:
        for w in ws:
            counts[w.lower()] = counts.get(w.lower(), 0) + 1
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w, c in counts.items():
        if c >= 3:
            vocab[w] = len(vocab)
    D = {"vocab": vocab, "w2v": w2v, "train": tensors(train_bio, vocab), "test": tensors(test_bio, vocab)}
    model, losses = fit(D)
    ablation = []
    for name, sh, dr in [("word embedding only", False, False), ("+ word shape features", True, False)]:
        ablation.append({"Configuration": name, "F1": evaluate(fit(D, sh, dr)[0], D)[-1]["F1"]})
    report = evaluate(model, D)
    ablation.append({"Configuration": "+ shape + word dropout (used)", "F1": report[-1]["F1"]})
    return {"model": model, "vocab": vocab, "losses": losses, "report": report, "ablation": ablation,
            "bio_example": list(zip(*train_bio[0]))[:6], "n_train": len(train_bio), "n_test": len(test_bio),
            "oov_test": sum(w.lower() not in vocab for ws, _ in test_bio for w in ws),
            "w2v_init": sum(w in w2v for w in vocab),
            "params": sum(p.numel() for p in model.parameters())}


def detect(A4, text):
    """Detector 3 : tag every sentence, decode entities, vote = 0.70 x mean tag probability."""
    m, vocab, ents = A4["model"], A4["vocab"], []
    chunks = [t[c:c + MAXLEN] for t in token_sentences(text) for c in range(0, len(t), MAXLEN)]
    for toks in chunks:                              # padded exactly as in training
        x, s, _ = encode([w for w, _, _ in toks], ["O"] * len(toks), vocab)
        with torch.no_grad():
            prob, idx = m(torch.tensor([x]), torch.tensor([s])).softmax(-1)[0, :len(toks)].max(-1)
        word = lambda i: any(c.isalnum() for c in toks[i][0])
        for a, b, lab in spans_from_bio([TAGS[i] for i in idx.tolist()]):
            while a < b and not word(a):                     # trim punctuation at the edges
                a += 1
            while b > a and not word(b - 1):
                b -= 1
            if a == b:
                continue
            st, en = toks[a][1], toks[b - 1][2]
            ents.append({"text": text[st:en], "start": st, "end": en, "label": lab,
                         "src": ["bilstm"], "sc": [round(0.70 * prob[a:b].mean().item(), 3)],
                         "unseen": all(w.lower() not in vocab for w, _, _ in toks[a:b])})
    return ents
