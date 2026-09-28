"""Builds the AGORA index the dashboard uses: the tag model and the lexical (TF-IDF) indexes.

The app runs this automatically the first time it starts (about a minute), so you normally never
run it yourself. To rebuild by hand:

    python build_artifacts.py
"""
import argparse
import json
import os
import shutil
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier

import analyzer as az

ARTIFACT_VERSION = 2            # bump when the artifact format changes; the app rebuilds older ones
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "artifacts")


def find_data():
    """The AGORA CSVs, next to this file or one folder up (the repo root)."""
    for d in (os.path.join(HERE, "agora_dataset"), os.path.join(os.path.dirname(HERE), "agora_dataset")):
        if os.path.exists(os.path.join(d, "segments.csv")):
            return d
    raise FileNotFoundError("Couldn't find the agora_dataset folder next to the app or one folder up.")


def build(data=None, out=DEFAULT_OUT, log=print, min_support=20):
    """Train the tag model and build the lexical indexes into `out`. Takes about a minute."""
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    data = data or find_data()
    t0 = time.time()
    step = lambda msg: log(f"[{time.time() - t0:3.0f}s] {msg}")

    tmp = out + ".building"                          # write elsewhere first, so a half-built index is never used
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    # --- Data ----------------------------------------------------------------------------------
    documents = pd.read_csv(f"{data}/documents.csv")
    segments = pd.read_csv(f"{data}/segments.csv")
    segments["Text"] = segments["Text"].fillna("")
    tag_cols = [c for c in segments.columns if c.split(": ")[0] in az.DOMAINS]
    step(f"Loaded {len(documents):,} documents and {len(segments):,} segments")

    seg_text = (segments.rename(columns={"Document ID": "doc_id", "Segment position": "position", "Text": "text"})
                [["doc_id", "position", "text"]].sort_values(["doc_id", "position"]).reset_index(drop=True))
    seg_text.to_parquet(f"{tmp}/segments_text.parquet", index=False)

    docs = (seg_text.groupby("doc_id")["text"].apply(" ".join).rename("text").to_frame()
            .join(documents.set_index("AGORA ID")[["Official name", "Casual name", "Authority", "Most recent activity",
                                                   "Proposed date", "Link to document", "Short summary", "Tags"]]))
    docs.drop(columns="text").to_parquet(f"{tmp}/documents.parquet")

    # --- Tag classifier: TF-IDF n-grams + logistic regression ---------------------------------
    labeled = segments[segments["Segment annotated"]].copy()
    labeled[tag_cols] = az.add_parents(labeled[tag_cols])
    tags = [t for t in tag_cols if labeled[t].sum() >= min_support]
    Y = labeled[tags].to_numpy().astype(int)
    groups = labeled["Document ID"].to_numpy()

    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", token_pattern=az.TOKEN,
                            min_df=3, max_df=0.8, sublinear_tf=True, max_features=100_000)
    X = tfidf.fit_transform(labeled["Text"])
    make_model = lambda: OneVsRestClassifier(LogisticRegression(C=10, class_weight="balanced",
                                                                solver="liblinear", max_iter=1000))

    # Out-of-fold probabilities (grouped by document) give per-tag thresholds and honest reliability scores
    oof = np.zeros(Y.shape)
    for i, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, Y, groups), 1):
        oof[va] = make_model().fit(X[tr], Y[tr]).predict_proba(X[va])
        step(f"Tag model cross-validation {i}/5")
    grid = np.round(np.arange(0.05, 0.96, 0.025), 3)
    pred = oof[:, :, None] >= grid
    Yb = Y.astype(bool)[:, :, None]
    tp, fp, fn = (pred & Yb).sum(0), (pred & ~Yb).sum(0), (~pred & Yb).sum(0)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)

    model = make_model().fit(X, Y)
    for est in model.estimators_:                   # float32 weights: half the file size, same predictions
        if hasattr(est, "coef_"):
            est.coef_, est.intercept_ = est.coef_.astype(np.float32), est.intercept_.astype(np.float32)
    joblib.dump({"tfidf": tfidf, "model": model, "thresholds": grid[f1.argmax(axis=1)], "tags": tags,
                 "reliability": {t: round(float(f1[j].max()), 2) for j, t in enumerate(tags)},
                 "n_segments": len(labeled), "n_documents": int(labeled["Document ID"].nunique())},
                f"{tmp}/tagger.joblib", compress=3)
    step(f"Tag model trained ({len(tags)} tags)")

    # --- Lexical indexes: whole documents (similar laws) and segments (keyword search) ---------
    doc_vec = TfidfVectorizer(analyzer=az.phrases, min_df=2, max_df=0.5, sublinear_tf=True)
    X_docs = doc_vec.fit_transform(docs["text"]).astype(np.float32)
    joblib.dump(doc_vec, f"{tmp}/doc_vectorizer.joblib", compress=3)
    sp.save_npz(f"{tmp}/doc_matrix.npz", X_docs)
    np.save(f"{tmp}/doc_ids.npy", docs.index.to_numpy())
    sp.save_npz(f"{tmp}/segment_matrix.npz", doc_vec.transform(seg_text["text"]).astype(np.float32))
    step(f"Lexical indexes built ({X_docs.shape[1]:,} phrases)")

    json.dump({"version": ARTIFACT_VERSION, "scikit-learn": sklearn.__version__,
               "built": time.strftime("%Y-%m-%d %H:%M"), "agora_documents": int(len(docs))},
              open(f"{tmp}/manifest.json", "w"), indent=2)
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    step("Done")


def is_current(out=DEFAULT_OUT):
    """True if `out` holds a complete index in the current format."""
    try:
        with open(os.path.join(out, "manifest.json")) as f:
            return json.load(f).get("version") == ARTIFACT_VERSION
    except (OSError, ValueError):
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="folder with the AGORA CSVs (found automatically)")
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()
    build(args.data, args.out)
