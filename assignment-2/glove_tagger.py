"""
AME 5151 Applied Linear Algebra Lab - Assignment 2
Semantic tagging of Manipal LinkedIn text with GloVe (glove-wiki-gigaword-50).

Usage:
    GLOVE_PATH=path/to/glove_50_fast.wordvectors python glove_tagger.py [text_file]
"""

import os
import string
import sys
import unicodedata

import numpy as np
from gensim.models import KeyedVectors

from vec import Vec

GLOVE_PATH = os.environ.get("GLOVE_PATH", "assets/glove50/glove_50_fast.wordvectors")
TEXT_FILE = "manipal_linkedin.txt"
DIM = 50
TOLERANCE = 1e-9   # matrices are computed in float64, so a tight tolerance is fine
TOP_K = 8

# ---------------------------------------------------------------- Part A: tags
# Two multi-word tags: "artificial intelligence" and "public health".
TAG_VOCAB = (
    "research",
    "innovation",
    "education",
    "university",
    "students",
    "faculty",
    "campus",
    "engineering",
    "medicine",
    "public health",
    "technology",
    "artificial intelligence",
    "curriculum",
    "collaboration",
    "publication",
    "scholarship",
    "internship",
    "startup",
    "alumni",
    "careers",
)

# ---------------------------------------------------------- Part C: stopwords
STOP_WORDS = frozenset("""
the a an and or of to in on for is are was were with at by from
it its this that these those we our ours us you your i me my he she his her
they them their be been being am has have had do does did will would can
could should may might must as into about over after before during between
but so than then also very just not no nor all any each every some such
more most other own what which who whom when where why how here there
up out off again further once
http https www
""".split())

# string.punctuation only covers ASCII, so add common Unicode punctuation
# that shows up in copied LinkedIn posts (curly quotes, dashes, bullets ...).
PUNCTUATION = string.punctuation + "‘’“”–—•…"
PUNCT_TABLE = str.maketrans({ch: " " for ch in PUNCTUATION})


def load_glove(path=GLOVE_PATH):
    return KeyedVectors.load(path, mmap="r")


# ------------------------------------------------------ Part C: preprocessing
def lowercase(text):
    # NFKC first turns LinkedIn's styled bold letters into plain ones
    return unicodedata.normalize("NFKC", text).lower()


def strip_punctuation(text):
    cleaned = text.translate(PUNCT_TABLE)
    # drop anything that is still not a letter, digit or whitespace (emojis etc.)
    return "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in cleaned)


def tokenize(text):
    return text.split()


def drop_stop_words(tokens):
    return [t for t in tokens if t not in STOP_WORDS]


def drop_short(tokens, min_len=3):
    return [t for t in tokens if len(t) >= min_len]


def preprocess_text(text):
    """Steps 1-5 in order. Returns the list x_1..x_m."""
    return drop_short(drop_stop_words(tokenize(strip_punctuation(lowercase(text)))))


# ------------------------------------------------------- Part D: matrices
def build_tag_matrix(model, tags):
    """
    Returns (tag_names, T) with T of shape (20, 50), dtype float64.
    Multi-word tags are the mean of their component vectors. Any component
    word missing from the vocabulary raises a KeyError-like ValueError.
    """
    tags = list(tags)
    if len(tags) != 20:
        raise ValueError(f"Need exactly 20 tags, received {len(tags)}")

    T = np.zeros((len(tags), DIM), dtype=np.float64)
    for j, tag in enumerate(tags):
        parts = tag.split()
        missing = [p for p in parts if p not in model.key_to_index]
        if missing:
            raise ValueError(f"Tag not in vocabulary: {tag} (unknown: {', '.join(missing)})")
        k = len(parts)
        for p in parts:
            T[j] += model[p].astype(np.float64)
        T[j] /= k
    return tags, T


def build_text_matrix(model, tokens):
    """
    Returns (in_vocab_tokens, oov_tokens, W) where W has shape (n, 50),
    dtype float64. Order and repetitions of in-vocabulary tokens are kept.
    """
    in_vocab_tokens, oov_tokens = [], []
    for tok in tokens:
        (in_vocab_tokens if tok in model.key_to_index else oov_tokens).append(tok)

    W = np.zeros((len(in_vocab_tokens), DIM), dtype=np.float64)
    for i, tok in enumerate(in_vocab_tokens):
        W[i] = model[tok]
    return in_vocab_tokens, oov_tokens, W


# ------------------------------------------- Part E: Vec-class similarity
def similarity_matrix_vector_class(W, T):
    """Every entry is Vec(W[i]).cosine_similarity(Vec(T[j])); loops only."""
    word_vecs = [Vec(map(float, W[i])) for i in range(W.shape[0])]
    tag_vecs = [Vec(map(float, T[j])) for j in range(T.shape[0])]
    return [[w.cosine_similarity(t) for t in tag_vecs] for w in word_vecs]


# ---------------------------------------------- Part F: NumPy similarity
def row_normalize(M):
    norms = np.sqrt(np.sum(M * M, axis=1))[:, np.newaxis]
    if (norms == 0).any():
        raise ValueError("Cannot normalize a zero row")
    return M / norms


def similarity_matrix_numpy(W, T):
    """S = W_hat T_hat^T via a single matrix multiplication."""
    return np.matmul(row_normalize(W), row_normalize(T).T)


# ----------------------------------------------- Part H: max pooling
def rank_tags(tag_names, S, in_vocab_tokens):
    """
    Returns a list of dicts {tag, score, word} sorted by score (descending),
    where score = max over the tag's column and word is the arg-max token.
    """
    S = np.asarray(S)
    results = []
    for j, tag in enumerate(tag_names):
        column = S[:, j]
        i_best = int(column.argmax())
        results.append({"tag": tag, "score": float(column[i_best]),
                        "word": in_vocab_tokens[i_best]})
    return sorted(results, key=lambda r: -r["score"])


def show_ranking(ranking, k=TOP_K):
    header = f"{'Rank':<6}{'Tag':<26}{'Score':<9}{'Best-matching text word':<25}Match"
    print(header)
    print("-" * len(header))
    for pos, r in enumerate(ranking[:k], 1):
        kind = "same word" if r["word"] == r["tag"] else "related word"
        print(f"{pos:<6}{r['tag']:<26}{r['score']:<9.3f}{r['word']:<25}{kind}")
    same = sum(r["word"] == r["tag"] for r in ranking[:k])
    print(f"\n{same} of the top {k} tags were matched by the identical word, "
          f"{k - same} by a different word.")


# ------------------------------------------------------------------ main
def main(text_path=TEXT_FILE):
    with open(text_path, encoding="utf-8") as fh:
        raw = fh.read()

    model = load_glove()
    tokens = preprocess_text(raw)
    tag_names, T = build_tag_matrix(model, TAG_VOCAB)
    in_vocab, oov, W = build_text_matrix(model, tokens)

    print("Token statistics")
    print("  before preprocessing   :", len(raw.split()))
    print("  after preprocessing    :", len(tokens))
    print("  in-vocabulary          :", len(in_vocab))
    print("  out-of-vocabulary      :", len(oov))
    print("  distinct OOV tokens    :", ", ".join(sorted(set(oov))) or "(none)")
    print()
    print("T.shape:", T.shape)
    print("W.shape:", W.shape)
    print()

    if len(in_vocab) == 0:
        sys.exit("No in-vocabulary words left after preprocessing - "
                 "nothing to compare with the tags.")

    S_vector = np.array(similarity_matrix_vector_class(W, T))
    S_numpy = similarity_matrix_numpy(W, T)

    print("S.shape:", S_numpy.shape)
    print(f"np.allclose(S_vector, S_numpy, atol={TOLERANCE}):",
          np.allclose(S_vector, S_numpy, rtol=0, atol=TOLERANCE))
    print("np.max(np.abs(S_vector - S_numpy)):", np.max(np.abs(S_vector - S_numpy)))
    print()

    ranking = rank_tags(tag_names, S_numpy, in_vocab)
    show_ranking(ranking)

    # Exact matches always score 1, so also look at the closest *other* word.
    print("\nIgnoring exact matches, the closest text word for each top tag is:")
    tokens_arr = np.array(in_vocab)
    for r in ranking[:TOP_K]:
        j = tag_names.index(r["tag"])
        column = np.where(tokens_arr == r["tag"], -np.inf, S_numpy[:, j])
        i = int(column.argmax())
        print(f"  {r['tag']:<26}{in_vocab[i]:<20}{column[i]:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
