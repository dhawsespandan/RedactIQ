"""Phase 1 (Assignment 1) - Text preprocessing with NLTK.

Split the document into sentences and words, grab the formatted PII (email,
phone, PAN, date) BEFORE cleaning, then clean with regular expressions, remove
stop words, and compare Porter stemming with WordNet lemmatization.
Lesson used by every later phase: cleaning destroys PII, so detection always
runs on the tidied ORIGINAL text, never on the cleaned text.
"""
import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

for pkg, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                  ("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet.zip")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

# Patterns used to grab formatted PII before cleaning
PII_PATTERNS = {
    "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "PHONE": r"(?:\+91[-\s]?)?[6-9]\d{9}",
    "PAN":   r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "DATE":  r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
}
# Span tokenizer shared by phases 3 and 4: e-mails and phone numbers stay one token
TOKEN_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+|\+?\d+(?:-\d+)+|\+\d+|\w+|[^\w\s]")


def tidy(text):
    """Light normalisation: unify line endings and spaces, keep case and every PII character."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ").replace(" ", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@lru_cache(maxsize=16)
def tokenize(text):
    """(word, start, end) for every token, so a token can be mapped back to the text."""
    return tuple((m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text))


def token_sentences(text):
    """Tokens grouped into sentences; a sentence ends at . ! ? or at a line break."""
    toks, sents, cur = tokenize(text), [], []
    for i, t in enumerate(toks):
        cur.append(t)
        if t[0] in ".!?" or i + 1 == len(toks) or "\n" in text[t[2]:toks[i + 1][1]]:
            sents.append(cur)
            cur = []
    return sents


def clean(t):
    """Remove URLs, digits, punctuation and extra spaces using regex."""
    t = re.sub(r"https?://\S+", " ", t)      # URLs
    t = re.sub(r"\d+", " ", t)               # numbers
    t = re.sub(r"[^\w\s]", " ", t)           # punctuation
    return re.sub(r"\s+", " ", t).strip()    # extra whitespace


def lemma(wnl, w):
    """WordNet lemma: try the word as a verb first, then as a noun."""
    v = wnl.lemmatize(w, pos="v")
    return v if v != w else wnl.lemmatize(w)


def preprocess(text):
    """Run the whole Assignment 1 pipeline on one document and return every intermediate result."""
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text)
    pii = [(lab, m.start(), m.end(), m.group())                    # BEFORE cleaning
           for lab, pat in PII_PATTERNS.items() for m in re.finditer(pat, text)]
    cleaned = clean(text)
    words = [w.lower() for w in word_tokenize(cleaned) if w.isalpha()]
    stop = set(stopwords.words("english"))
    filtered = [w for w in words if w not in stop]
    ps, wnl = PorterStemmer(), WordNetLemmatizer()
    norm = [(w, ps.stem(w), lemma(wnl, w)) for w in dict.fromkeys(filtered)]
    return {
        "sentences": sentences, "tokens": tokens, "pii": pii, "cleaned": cleaned,
        "survived": sum(p[3] in cleaned for p in pii),            # PII still intact after cleaning
        "words": words, "filtered": filtered, "norm": norm,
        "differ": [r for r in norm if r[1] != r[2]],               # stem and lemma disagree
    }
