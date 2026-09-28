# AI Bill Analyzer

A Streamlit dashboard that analyzes a U.S. bill (congress.gov / govinfo XML) against the
[AGORA](https://doi.org/10.5281/zenodo.21964209) dataset of AI laws:

| Tab | What it shows |
|---|---|
| Overview | Likely earlier version of the bill in AGORA and what changed, main actors, predicted AGORA taxonomy, numbering gaps |
| Provisions | Full outline (text inserted into other laws marked in blue), definitions, cross-references, named statutes |
| Legal effect | Obligations, permissions, prohibitions; Hohfeldian positions; conditions, exceptions, deadlines, hedges |
| Similar laws | Closest AGORA documents by meaning (embeddings) and wording (TF-IDF), matching passages, version comparison, free-text search |
| AGORA tags | Predicted tags with evidence phrases, per-tag model reliability, and where in the bill each tag comes from |
| Phrases | Most frequent and most distinctive 1–3-word phrases |

## Files

```
app.py               the dashboard (UI only)
analyzer.py          all analysis functions, shared by the app and the build script
build_artifacts.py   one-time: trains the tag model and precomputes the AGORA indexes into artifacts/
requirements.txt
agora_dataset/       AGORA CSVs (already in the repo)
sample_bill/         H.R. 9334, shown when no file is uploaded
artifacts/           created by build_artifacts.py; commit it so the app never has to rebuild
```

## 1. Build the artifacts (once, ~5 minutes on a Colab GPU)

In a Colab notebook with a GPU runtime (Runtime → Change runtime type → T4 GPU):

```
!git clone https://github.com/cbirjandi/StressBill.git
%cd StressBill
!pip install -q -r requirements.txt
!python build_artifacts.py
!zip -r artifacts.zip artifacts
```

Download `artifacts.zip` from the Files panel, unzip it into the repo folder, and commit the `artifacts/`
folder (every file is under GitHub's 25 MB browser-upload limit).

You can also run `python build_artifacts.py` on your own computer: same result, but the embedding step
takes 10–15 minutes on a CPU.

**Use the same scikit-learn version to build and to run** (pinned in `requirements.txt`). Saved models
from one version may not load in another.

## 2. Run the app locally

```
pip install -r requirements.txt
streamlit run app.py
```

It opens in your browser. The first load takes a few seconds (plus a one-time download of the
~90 MB embedding model); after that each bill is analyzed in a few seconds.

## 3. Share it online (optional)

Push the repo to GitHub, then at [share.streamlit.io](https://share.streamlit.io) create an app pointing at
`app.py` (choose Python 3.12 under Advanced settings). Anyone with the link can use it.

If the cloud build fails or runs out of memory, the likely cause is PyTorch (pulled in by
`sentence-transformers`). The lighter fallback: rebuild with `python build_artifacts.py --embedder lsa`,
delete the `sentence-transformers` line from `requirements.txt`, and commit the new `artifacts/`.
Semantic matching becomes less accurate, but everything else is unchanged.

## Notes

- The tag model is the notebook's model A (TF-IDF n-grams + logistic regression), refit on all 5,647
  human-annotated AGORA segments, with per-tag thresholds and reliability scores from 5-fold
  cross-validation grouped by document.
- The deontic extraction fixes a bug in the notebook's section 8, where `action` was only filled in
  when spaCy failed to find a subject.
- Data: Arnold, Z., Melot, J., Enwereazu, O., Schiff, D. S., Schiff, K. J., & Girard, T. (2026).
  AGORA Dataset (Version 1.32.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.21964209
