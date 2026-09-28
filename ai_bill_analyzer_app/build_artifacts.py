"""One-time build: train the tag model and precompute the AGORA indexes the dashboard loads.

Run from the repo folder (Colab with a GPU is fastest for the embeddings):

    python build_artifacts.py                      # sentence-transformer embeddings (recommended)
    python build_artifacts.py --embedder lsa       # no PyTorch needed; lighter but less accurate

Takes ~2 minutes plus embedding time (~1 min on a GPU, ~10-15 min on a CPU).
Writes everything to artifacts/ (about 70-100 MB).
"""
import argparse
import json
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.multiclass import OneVsRestClassifier

import analyzer as az

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

p = argparse.ArgumentParser()
p.add_argument("--data", default="agora_dataset")
p.add_argument("--out", default="artifacts")
p.add_argument("--embedder", choices=["minilm", "lsa"], default="minilm")
p.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
p.add_argument("--min-support", type=int, default=20)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()
log = lambda msg: print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

# --- Data --------------------------------------------------------------------------------------
documents = pd.read_csv(f"{args.data}/documents.csv")
segments = pd.read_csv(f"{args.data}/segments.csv")
segments["Text"] = segments["Text"].fillna("")
TAG_COLS = [c for c in segments.columns if c.split(": ")[0] in az.DOMAINS]
log(f"{len(documents):,} documents, {len(segments):,} segments, {len(TAG_COLS)} tags")

seg_text = (segments.rename(columns={"Document ID": "doc_id", "Segment position": "position", "Text": "text"})
            [["doc_id", "position", "text"]].sort_values(["doc_id", "position"]).reset_index(drop=True))
seg_text.to_parquet(f"{args.out}/segments_text.parquet", index=False)

docs = (seg_text.groupby("doc_id")["text"].apply(" ".join).rename("text").to_frame()
        .join(documents.set_index("AGORA ID")[["Official name", "Casual name", "Authority", "Most recent activity",
                                               "Proposed date", "Link to document", "Short summary", "Tags"]]))
docs.drop(columns="text").to_parquet(f"{args.out}/documents.parquet")

# --- Tag classifier (model A: TF-IDF n-grams + logistic regression) ---------------------------
labeled = segments[segments["Segment annotated"]].copy()
labeled[TAG_COLS] = az.add_parents(labeled[TAG_COLS])
TAGS = [t for t in TAG_COLS if labeled[t].sum() >= args.min_support]
Y = labeled[TAGS].to_numpy().astype(int)
groups = labeled["Document ID"].to_numpy()

tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", token_pattern=az.TOKEN,
                        min_df=3, max_df=0.8, sublinear_tf=True, max_features=100_000)
X = tfidf.fit_transform(labeled["Text"])
make_model = lambda: OneVsRestClassifier(LogisticRegression(C=10, class_weight="balanced",
                                                            solver="liblinear", max_iter=1000))

# Out-of-fold probabilities, grouped by document, for thresholds and honest per-tag reliability
oof = np.zeros(Y.shape)
for tr, va in GroupKFold(n_splits=5).split(X, Y, groups):
    oof[va] = make_model().fit(X[tr], Y[tr]).predict_proba(X[va])
log("Cross-validation done")

GRID = np.round(np.arange(0.05, 0.96, 0.025), 3)
pred = oof[:, :, None] >= GRID
Yb = Y.astype(bool)[:, :, None]
tp, fp, fn = (pred & Yb).sum(0), (pred & ~Yb).sum(0), (~pred & Yb).sum(0)
f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
thresholds = GRID[f1.argmax(axis=1)]
reliability = {t: round(float(f1[j].max()), 2) for j, t in enumerate(TAGS)}

model = make_model().fit(X, Y)          # final model uses all labeled data
for est in model.estimators_:           # float32 weights halve the file size; predictions are unaffected in practice
    if hasattr(est, "coef_"):
        est.coef_, est.intercept_ = est.coef_.astype(np.float32), est.intercept_.astype(np.float32)
joblib.dump({"tfidf": tfidf, "model": model, "thresholds": thresholds, "tags": TAGS,
             "reliability": reliability, "n_segments": len(labeled),
             "n_documents": int(labeled["Document ID"].nunique())}, f"{args.out}/tagger.joblib", compress=3)
log(f"Tag model saved ({len(TAGS)} tags; median cross-val F1 {np.median(list(reliability.values())):.2f})")

# --- Lexical index (TF-IDF over whole documents) ----------------------------------------------
doc_vec = TfidfVectorizer(analyzer=az.phrases, min_df=2, max_df=0.5, sublinear_tf=True)
X_docs = doc_vec.fit_transform(docs["text"]).astype(np.float32)
joblib.dump(doc_vec, f"{args.out}/doc_vectorizer.joblib", compress=3)
sp.save_npz(f"{args.out}/doc_matrix.npz", X_docs)
np.save(f"{args.out}/doc_ids.npy", docs.index.to_numpy())
log(f"Lexical index saved ({X_docs.shape[1]:,} phrases)")

# --- Semantic index (chunk embeddings) --------------------------------------------------------
chunks = az.make_chunks(seg_text)
if args.embedder == "minilm":
    embed = az.sentence_transformer_embedder(args.model_name)
    spec = {"kind": "sentence-transformers", "model": args.model_name}
else:
    lsa_tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english", max_features=20_000)
    svd = TruncatedSVD(384, random_state=0).fit(lsa_tfidf.fit_transform(seg_text["text"]))
    embed = az.LSAEmbedder(lsa_tfidf, svd.components_)
    joblib.dump(embed, f"{args.out}/lsa_embedder.joblib", compress=3)
    spec = {"kind": "lsa", "model": "LSA-384"}
E = embed(chunks["text"]).astype(np.float16)
np.save(f"{args.out}/chunk_embeddings.npy", E)
log(f"Embeddings saved: {E.shape[0]:,} chunks x {E.shape[1]} ({spec['model']})")

json.dump({"embedder": spec, "n_chunks": int(len(chunks)), "scikit-learn": sklearn.__version__,
           "built": time.strftime("%Y-%m-%d %H:%M"), "agora_documents": int(len(docs))},
          open(f"{args.out}/manifest.json", "w"), indent=2)
log(f"Done. Folder size: {sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out)) / 1e6:.0f} MB")
