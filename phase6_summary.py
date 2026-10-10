"""Phase 6 (Assignment 6) - mini-BERT pre-trained with masked language modelling.

A compact BERT (2 layers, 4 heads, 96 dimensions) learns the language of audit
reports from scratch. Each row of this run's audit trail is rewritten as a
sentence in its vocabulary ("the regex rule redacted an email address with
high confidence"); sentences are embedded by mean-pooling, clustered with
k-means, and the sentence nearest each centroid forms the extractive summary.
"""
import random
import warnings

import torch
import torch.nn as nn
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

SUBJ = ["the detector", "the regex rule", "the ner model", "the bilstm tagger", "the fusion layer",
        "the audit report", "the confidence score", "the redaction stage"]
VERB = ["identified", "flagged", "redacted", "reviewed", "scored", "matched", "logged"]
OBJ = ["an email address", "a phone number", "a pan number", "a person name",
       "an organisation name", "an aadhaar number"]
TAIL = ["with high confidence", "with low confidence", "for manual review",
        "in the input document", "before translation", "during the audit"]

PAD, CLS, SEP, MASK, UNK = 0, 1, 2, 3, 4
vocab = {"[PAD]": PAD, "[CLS]": CLS, "[SEP]": SEP, "[MASK]": MASK, "[UNK]": UNK}
for phrase in SUBJ + VERB + OBJ + TAIL:
    for w in phrase.split():
        vocab.setdefault(w, len(vocab))
inv = {i: w for w, i in vocab.items()}
MAXLEN = 12


def encode(s):
    """Every sequence is wrapped as [CLS] ... [SEP], exactly as in BERT."""
    ids = [CLS] + [MASK if w == "[MASK]" else vocab.get(w, UNK) for w in s.split()][:MAXLEN - 2] + [SEP]
    return ids + [PAD] * (MAXLEN - len(ids))


class MiniBERT(nn.Module):
    """Token + learned position embeddings -> transformer encoder layers -> masked LM head."""
    def __init__(self, nvocab, d=96, heads=4, layers=2):
        super().__init__()
        self.tok = nn.Embedding(nvocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(MAXLEN, d)
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.mlm = nn.Linear(d, nvocab)              # masked language model head

    def forward(self, x):
        p = torch.arange(x.size(1)).unsqueeze(0).expand_as(x)
        h = self.enc(self.norm(self.tok(x) + self.pos(p)), src_key_padding_mask=(x == PAD))
        return h, self.mlm(h)


def mask_batch(x, p=0.15):
    """Replace 15 percent of real tokens by [MASK]; labels are -100 everywhere else."""
    y = torch.full_like(x, -100)
    sel = (torch.rand_like(x, dtype=torch.float) < p) & (x != PAD) & (x != CLS) & (x != SEP)
    y[sel] = x[sel]
    xm = x.clone()
    xm[sel] = MASK
    return xm, y


def train(epochs=15, seed=42):
    """Pre-train the mini-BERT with the masked language modelling objective."""
    random.seed(seed)
    torch.manual_seed(seed)
    corpus = [f"{random.choice(SUBJ)} {random.choice(VERB)} {random.choice(OBJ)} {random.choice(TAIL)}"
              for _ in range(4000)]
    X = torch.tensor([encode(s) for s in corpus])
    model = MiniBERT(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    losses = []
    for _ in range(epochs):
        model.train()
        tot = n = 0
        for i in range(0, len(X), 64):
            xm, y = mask_batch(X[i:i + 64])
            if (y != -100).sum() == 0:               # no token was masked
                continue
            opt.zero_grad()
            loss = lossf(model(xm)[1].reshape(-1, len(vocab)), y.reshape(-1))
            loss.backward()
            opt.step()
            tot, n = tot + loss.item() * len(xm), n + len(xm)
        losses.append(tot / n)
    model.eval()
    fills = [(s, fill_mask(model, s)) for s in ["the detector [MASK] an email address with high confidence",
                                               "the regex rule matched a [MASK] number in the input document"]]
    cos = nn.CosineSimilarity(dim=0)(token_vec(model, "the detector flagged a person name for manual review", "name"),
                                     token_vec(model, "the ner model scored an organisation name during the audit", "name"))
    return {"model": model, "losses": losses, "fills": fills, "cosine": round(cos.item(), 3),
            "params": sum(p.numel() for p in model.parameters()), "vocab": len(vocab)}


def fill_mask(model, sent):
    """Three highest-scoring words for the [MASK] position."""
    ids = encode(sent)
    with torch.no_grad():
        logits = model(torch.tensor([ids]))[1]
    return [inv[i] for i in logits[0, ids.index(MASK)].topk(3).indices.tolist()]


def token_vec(model, sentence, word):
    """Contextual vector of one word inside one sentence."""
    ids = encode(sentence)
    with torch.no_grad():
        return model(torch.tensor([ids]))[0][0, ids.index(vocab[word])]


def embed(model, sentence):
    """Mean-pool the final hidden states over the real tokens."""
    ids = torch.tensor([encode(sentence)])
    with torch.no_grad():
        h = model(ids)[0]
    m = (ids != PAD).unsqueeze(-1).float()
    return ((h * m).sum(1) / m.sum(1)).squeeze(0)


DETECTOR = {"regex": "the regex rule", "spacy": "the ner model", "bilstm": "the bilstm tagger"}
OBJECT = {"EMAIL": "an email address", "PHONE": "a phone number", "PAN": "a pan number",
          "PERSON": "a person name", "ORG": "an organisation name", "AADHAAR": "an aadhaar number"}
DECISION = {"AUTO_REDACT": ("redacted", "with high confidence"),
            "REVIEW": ("flagged", "for manual review"), "LOG_ONLY": ("logged", "with low confidence")}


def describe(e):
    """One audit-trail row -> one sentence in the mini-BERT vocabulary."""
    srcs = list(dict.fromkeys(e["src"]))
    subj = DETECTOR[srcs[0]] if len(srcs) == 1 else "the fusion layer"
    verb, tail = DECISION[e["band"]]
    return f"{subj} {verb} {OBJECT[e['label']]} {tail}"


def summarise(model, sentences, k=3):
    """k-means over the sentence embeddings; keep the sentence nearest each centroid, in order."""
    k = min(k, len(set(sentences)))
    if k <= 1:                                       # nothing to cluster
        return [0] if sentences else [], [0] * len(sentences)
    V = torch.stack([embed(model, s) for s in sentences]).numpy()
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(V)
    picked = []
    for c in range(k):
        d = ((V - km.cluster_centers_[c]) ** 2).sum(1)
        d[km.labels_ != c] = 1e9
        picked.append(int(d.argmin()))
    return sorted(picked), km.labels_.tolist()
