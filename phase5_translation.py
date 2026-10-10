"""Phase 5 (Assignment 5) - Seq2seq with Bahdanau attention, English -> romanised Hindi.

The model is trained only on DELEXICALISED sentence pairs (<name>, <email>, ...),
so no identifier ever enters its vocabulary. The app translates the REDACTED
document: each [TYPE_REDACTED] tag becomes a placeholder, the sentence is
translated, and the tag is put back. A sentence is marked "not translated" when
it is outside the translator's vocabulary or the placeholders do not come back
unchanged, and the output is checked for leaks.
"""
import random
import re

import torch
import torch.nn as nn

from phase1_preprocessing import sent_tokenize

# ---------------- Parallel corpus (English -> romanised Hindi) ----------------
TEMPLATES = [
    ("please contact {name} today", "kripaya aaj {name} se sampark karein"),
    ("the email of {name} is {email}", "{name} ka email {email} hai"),
    ("the mobile number of {name} is {phone}", "{name} ka mobile number {phone} hai"),
    ("the pan of {name} is {pan}", "{name} ka pan {pan} hai"),
    ("{name} works at the office", "{name} karyalay mein kaam karta hai"),
    ("the meeting is scheduled next week", "baithak agle saptah niyat hai"),
    ("please submit the report today", "kripaya aaj report jama karein"),
    ("please contact {name} at {email} or on {phone}",
     "kripaya {name} se {email} ya {phone} par sampark karein"),
    ("employee {name} of {org} has pan {pan}", "{org} ke karmachari {name} ka pan {pan} hai"),
    ("ticket raised by {name}", "{name} dwara ticket uthaya gaya"),
    ("registered mobile is {phone}", "panjikrit mobile {phone} hai"),
    ("the applicant {name} submitted pan {pan} for verification",
     "aavedak {name} ne satyapan ke liye pan {pan} jama kiya"),
    ("aadhaar {aadhaar} belongs to {name}", "aadhaar {aadhaar} {name} ka hai"),
    ("the compliance review meeting is scheduled next week",
     "anupalan samiksha baithak agle saptah niyat hai"),
]
NAMES = ["rohan mehta", "anita desai", "kabir rao", "priya sharma"]
ORGS = ["infosys limited", "wipro technologies"]
PLACEHOLDER = {"PERSON": "<name>", "EMAIL": "<email>", "PHONE": "<phone>",
               "PAN": "<pan>", "ORG": "<org>", "AADHAAR": "<aadhaar>"}
TAG_RE = re.compile(r"\[([A-Z]+)_REDACTED\]")


def digits(k):
    return "".join(random.choices("0123456789", k=k))


def sample(delex):
    """Fill one template. delex=True keeps the placeholders instead of real values."""
    en, hi = random.choice(TEMPLATES)
    n = random.choice(NAMES)
    vals = {"name": n, "email": n.replace(" ", ".") + "@sitpune.edu.in",
            "phone": str(random.randint(6, 9)) + digits(9), "org": random.choice(ORGS),
            "pan": "".join(random.choices("ABCDEFGH", k=5)) + digits(4) + "A",
            "aadhaar": " ".join(digits(4) for _ in range(3))}
    if delex:
        vals = {k: f"<{k}>" for k in vals}
    return en.format(**vals), hi.format(**vals)


# ---------------- Vocabularies and encoding ----------------
PAD, SOS, EOS, UNK = 0, 1, 2, 3
SRC_L = TGT_L = 14


def build_vocab(sents):
    v = {"<pad>": PAD, "<sos>": SOS, "<eos>": EOS, "<unk>": UNK}
    for s in sents:
        for w in s.split():
            v.setdefault(w, len(v))
    return v


def encode(s, v, n, eos=False):
    ids = [v.get(w, UNK) for w in s.split()][:n - 1]
    if eos:
        ids = ids + [EOS]
    return ids + [PAD] * (n - len(ids))


# ---------------- Encoder-decoder with Bahdanau attention ----------------
class Encoder(nn.Module):
    """Reads the source sentence and returns one hidden state per token."""
    def __init__(self, n, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(n, emb, padding_idx=PAD)
        self.rnn = nn.GRU(emb, hid, batch_first=True)

    def forward(self, x):
        return self.rnn(self.emb(x))                 # all states, final state


class Attention(nn.Module):
    """Bahdanau attention: score every encoder state against the decoder state."""
    def __init__(self, hid=128):
        super().__init__()
        self.W = nn.Linear(hid * 2, hid)
        self.v = nn.Linear(hid, 1, bias=False)

    def forward(self, dec_h, enc_out, mask):
        d = dec_h.unsqueeze(1).expand(-1, enc_out.size(1), -1)
        e = self.v(torch.tanh(self.W(torch.cat([d, enc_out], -1)))).squeeze(-1)
        a = torch.softmax(e.masked_fill(~mask, -1e4), dim=1)    # never attend to padding
        return torch.bmm(a.unsqueeze(1), enc_out).squeeze(1), a


class Decoder(nn.Module):
    """Generates the target one word at a time from the context vector."""
    def __init__(self, n, emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(n, emb, padding_idx=PAD)
        self.attn = Attention(hid)
        self.rnn = nn.GRU(emb + hid, hid, batch_first=True)
        self.fc = nn.Linear(hid * 2, n)

    def step(self, y, h, enc_out, mask):
        ctx, a = self.attn(h[-1], enc_out, mask)
        o, h = self.rnn(torch.cat([self.emb(y).unsqueeze(1), ctx.unsqueeze(1)], -1), h)
        return self.fc(torch.cat([o.squeeze(1), ctx], -1)), h, a


def train(epochs=10, seed=42):
    """Train on delexicalised pairs with teacher forcing; also compare vocabulary sizes."""
    random.seed(seed)
    torch.manual_seed(seed)
    lex = [sample(False) for _ in range(3000)]
    delex = [sample(True) for _ in range(3000)]
    sizes = {name: (len(build_vocab([s for s, _ in c])), len(build_vocab([t for _, t in c])))
             for name, c in [("lexicalised", lex), ("delexicalised", delex)]}
    sv, tv = build_vocab([s for s, _ in delex]), build_vocab([t for _, t in delex])
    X = torch.tensor([encode(s, sv, SRC_L) for s, _ in delex])
    Y = torch.tensor([[SOS] + encode(t, tv, TGT_L, eos=True)[:TGT_L - 1] for _, t in delex])
    enc, dec = Encoder(len(sv)), Decoder(len(tv))
    ps = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(ps, lr=2e-3)
    lossf = nn.CrossEntropyLoss(ignore_index=PAD)    # one loss over the flattened sequence
    losses = []
    for _ in range(epochs):
        enc.train(); dec.train()
        tot = 0.0
        for i in range(0, len(X), 64):
            xb, yb = X[i:i + 64], Y[i:i + 64]
            mask = xb != PAD
            enc_out, h = enc(xb)
            steps = []
            for t in range(1, yb.size(1)):           # teacher forcing: feed the true previous word
                logits, h, _ = dec.step(yb[:, t - 1], h, enc_out, mask)
                steps.append(logits)
            out = torch.stack(steps, 1)
            opt.zero_grad()
            loss = lossf(out.reshape(-1, out.size(-1)), yb[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ps, 1.0)  # guard against exploding gradients
            opt.step()
            tot += loss.item() * len(xb)
        losses.append(tot / len(X))
    return {"enc": enc.eval(), "dec": dec.eval(), "sv": sv, "tv": tv, "losses": losses, "sizes": sizes,
            "params": sum(p.numel() for p in ps), "pairs": delex[:2]}


def translate(A5, sentence):
    """Greedy decoding: take the highest-scoring word at each step and feed it back."""
    inv = {i: w for w, i in A5["tv"].items()}
    x = torch.tensor([encode(sentence, A5["sv"], SRC_L)])
    words = []
    with torch.no_grad():
        enc_out, h = A5["enc"](x)
        y = torch.tensor([SOS])
        for _ in range(TGT_L):
            logits, h, _ = A5["dec"].step(y, h, enc_out, x != PAD)
            y = logits.argmax(-1)
            if y.item() == EOS:
                break
            words.append(inv.get(y.item(), "<unk>"))
    return " ".join(words)


def translate_document(A5, redacted):
    """Translate a redacted document sentence by sentence; tags bypass the network."""
    rows, lines = [], []
    for line in redacted.split("\n"):
        out = []
        for sent in sent_tokenize(line) if line.strip() else []:
            src = TAG_RE.sub(lambda m: f" {PLACEHOLDER.get(m.group(1), '<unk>')} ", sent).lower()
            words = re.sub(r"[.,;:!?]", " ", src).split()
            ok = 0 < len(words) < SRC_L and all(w in A5["sv"] for w in words)
            hi = translate(A5, " ".join(words)) if ok else ""
            why = "" if ok else "not translated (outside vocabulary)"
            if ok and sorted(re.findall(r"<\w+>", hi)) != sorted(re.findall(r"<\w+>", src)):
                ok, why = False, "not translated (placeholders changed)"   # tags in must come out
            for tag, ph in PLACEHOLDER.items():      # restore the redaction tags
                hi = hi.replace(ph, f"[{tag}_REDACTED]")
            rows.append({"English (redacted)": sent, "Model input": " ".join(words) if ok else "",
                         "Hindi (romanised)": hi if ok else why})
            out.append(hi[:1].upper() + hi[1:] + "." if ok else f"[not translated] {sent}")
        lines.append(" ".join(out))
    return "\n".join(lines), rows


def leaks(output, secrets):
    """Every redacted value that still appears in the output (should be none)."""
    return [s for s in secrets if re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", output, re.I)]
