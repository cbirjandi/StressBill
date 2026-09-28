"""AI Bill Analyzer: Streamlit dashboard.

    streamlit run app.py

Needs the artifacts/ folder made by build_artifacts.py (run that once first).
"""
import html
import json
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

import analyzer as az

ART = "artifacts"
SAMPLE = os.path.join("sample_bill", "BILLS-119hr9334ih (1).xml")
PRIOR_VERSION_CUTOFF = 0.5          # lexical similarity above which a match is flagged as a likely earlier version

st.set_page_config(page_title="AI Bill Analyzer", page_icon="⚖️", layout="wide")


# ---------------------------------------------------------------------------------------------
# Loading (once per app process) and analysis (once per bill)
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading AGORA indexes…")
def load_index():
    manifest = json.load(open(f"{ART}/manifest.json"))
    tagger = joblib.load(f"{ART}/tagger.joblib")
    doc_ids = np.load(f"{ART}/doc_ids.npy")
    docs = pd.read_parquet(f"{ART}/documents.parquet").loc[doc_ids]
    seg_text = pd.read_parquet(f"{ART}/segments_text.parquet")
    chunks = az.make_chunks(seg_text)
    E = np.load(f"{ART}/chunk_embeddings.npy").astype(np.float32)
    if len(E) != len(chunks):
        raise RuntimeError("Embeddings don't match the segments file; re-run build_artifacts.py.")
    ids = chunks["doc_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    return {"manifest": manifest, "tagger": tagger, "docs": docs, "seg_text": seg_text,
            "doc_vec": joblib.load(f"{ART}/doc_vectorizer.joblib"), "X_docs": sp.load_npz(f"{ART}/doc_matrix.npz"),
            "chunks": chunks, "E": E, "doc_starts": starts, "doc_order": ids[starts]}


@st.cache_resource(show_spinner="Loading the embedding model…")
def load_embedder(kind, model_name):
    if kind == "lsa":
        return joblib.load(f"{ART}/lsa_embedder.joblib")
    return az.sentence_transformer_embedder(model_name)


@st.cache_resource(show_spinner="Loading spaCy…")
def load_nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def embedder():
    spec = load_index()["manifest"]["embedder"]
    return load_embedder(spec["kind"], spec["model"])


def doc_text(doc_id):
    st_ = load_index()["seg_text"]
    return " ".join(st_.loc[st_["doc_id"] == doc_id, "text"])


@st.cache_data(show_spinner="Analyzing the bill…", max_entries=10)
def analyze(xml_bytes):
    idx, embed = load_index(), embedder()
    soup = az.load_soup(xml_bytes)
    prov = az.parse_provisions(soup)
    segs = az.bill_segments(prov)
    bill_text = " ".join(segs["text"])

    # Similarity: lexical over whole documents, semantic section by section
    lexical = (idx["X_docs"] @ idx["doc_vec"].transform([bill_text]).T).toarray().ravel()
    substantive = segs[~segs["header"].str.contains(az.BOILERPLATE, case=False, regex=True)]
    if substantive.empty:
        substantive = segs
    M = az.section_by_document(substantive["text"], embed, idx["E"], idx["doc_starts"])
    sem = pd.DataFrame({"coverage": M.mean(axis=0), "best": M.max(axis=0),
                        "best section": substantive["citation"].to_numpy()[M.argmax(axis=0)]},
                       index=idx["doc_order"])
    similar = (idx["docs"].assign(lexical=lexical).join(sem, how="inner")
               .sort_values("coverage", ascending=False))

    lex_rank = similar.sort_values("lexical", ascending=False)
    prior = None
    if lex_rank["lexical"].iloc[0] >= PRIOR_VERSION_CUTOFF:
        pid = lex_rank.index[0]
        prior = {"id": int(pid), "lexical": float(lex_rank["lexical"].iloc[0]),
                 "runner_up": float(lex_rank["lexical"].iloc[1]),
                 "compare": az.compare_versions(bill_text, doc_text(pid))}

    t = idx["tagger"]
    tags, probs = az.tag_segments(segs, t["tfidf"], t["model"], t["thresholds"], t["tags"], t["reliability"])

    return {"meta": az.metadata(soup), "prov": prov, "segs": segs, "substantive": substantive,
            "bill_text": bill_text, "gaps": az.numbering_gaps(prov), "defs": az.definitions(prov),
            "refs": az.references(soup), "acts": az.named_acts(prov),
            "rules": az.deontic_rules(prov, load_nlp()), "similar": similar, "prior": prior,
            "tags": tags, "probs": probs,
            "frequent": {n: az.top_ngrams(bill_text, n) for n in (1, 2, 3)},
            "distinctive": az.distinctive_ngrams(bill_text, idx["doc_vec"])}


@st.cache_data(show_spinner="Comparing texts…", max_entries=20)
def compare(bill_text, doc_id):
    return az.compare_versions(bill_text, doc_text(doc_id))


@st.cache_data(show_spinner="Finding matching passages…", max_entries=20)
def passages(substantive, doc_id):
    idx = load_index()
    return az.best_passages(substantive, doc_id, embedder(), idx["E"], idx["chunks"])


@st.cache_data(show_spinner="Searching AGORA…", max_entries=50)
def search(query, k=10):
    idx = load_index()
    scores = idx["E"] @ embedder()([query])[0].astype(np.float32)
    ch = idx["chunks"].assign(score=scores)
    best = ch.loc[ch.groupby(["doc_id", "position"])["score"].idxmax()].nlargest(k, "score")
    best["document"] = best["doc_id"].map(idx["docs"]["Casual name"])
    return best[["score", "document", "doc_id", "text"]]


# ---------------------------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------------------------
LINK = st.column_config.LinkColumn("Link", display_text="open")


def ngram_table(by_n, fmt):
    cols = {f"{n}-word": pd.Series([f"{p}  ({fmt.format(v)})" for p, v in s.items()]) for n, s in by_n.items()}
    out = pd.DataFrame(cols).fillna("")
    out.index += 1
    return out


def md(text):
    """Escape characters Streamlit markdown would otherwise interpret ($ starts LaTeX, * italics, ...)."""
    return re.sub(r"([\\`*_\[\]$~#<>|])", r"\\\1", text)


def show_changes(cmp, max_rows=12):
    c1, c2, c3 = st.columns(3)
    c1.metric("Earlier version", f"{cmp['old words']:,} words")
    c2.metric("This bill", f"{cmp['new words']:,} words")
    c3.metric("Earlier text carried over", f"{cmp['share of old kept']:.0%}")
    ch = cmp["changes"]
    st.caption(f"{cmp['share of new that is old']:.0%} of this bill's words come from passages shared with the "
               f"earlier text. {len(ch)} changes of 12+ words" + (f"; the {max_rows} largest:" if len(ch) > max_rows else ":"))
    for _, r in ch.head(max_rows).iterrows():
        preview = (r["added text"] or r["removed text"])[:90]
        with st.expander(f"{r['change'].capitalize()} · {r['words']} words · {preview}…"):
            if r["removed text"]:
                st.markdown(":red[**Removed:**] " + md(r["removed text"]))
            if r["added text"]:
                st.markdown(":green[**Added:**] " + md(r["added text"]))


def outline_html(rows):
    parts = []
    for _, r in rows.iterrows():
        head = f"<b>{html.escape(r['header'])}.</b> " if r["header"] else ""
        body = " ".join(x for x in [r["text"], r["continuation"]] if x)
        style = f"margin-left:{1.4 * r['depth']}em;margin-bottom:0.35em;"
        if r["quoted"]:
            style += "border-left:3px solid #8ab4f8;padding-left:0.6em;"
        parts.append(f"<div style='{style}'>{html.escape(r['enum'])} {head}{html.escape(body)}</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------------------------
# Sidebar: choose a bill
# ---------------------------------------------------------------------------------------------
st.sidebar.title("⚖️ AI Bill Analyzer")
if not os.path.exists(f"{ART}/manifest.json"):
    st.error("No `artifacts/` folder found. Run `python build_artifacts.py` once, then reload.")
    st.stop()

upload = st.sidebar.file_uploader("Upload bill XML (congress.gov / govinfo format)", type=["xml"])
if upload is not None:
    xml_bytes, source = upload.getvalue(), upload.name
elif os.path.exists(SAMPLE):
    with open(SAMPLE, "rb") as f:
        xml_bytes, source = f.read(), "Sample: H.R. 9334"
    st.sidebar.caption("No file uploaded, so showing the sample bill.")
else:
    st.info("Upload a bill's XML file in the sidebar to begin.")
    st.stop()

idx = load_index()
m = idx["manifest"]
st.sidebar.divider()
st.sidebar.caption(
    f"**Reference data:** AGORA dataset, {m['agora_documents']:,} documents. "
    f"Tag model trained on {idx['tagger']['n_segments']:,} human-annotated segments. "
    f"Embeddings: `{m['embedder']['model']}`. Built {m['built']}.")
st.sidebar.caption("Arnold, Z., Melot, J., Enwereazu, O., Schiff, D. S., Schiff, K. J., & Girard, T. (2026). "
                   "AGORA Dataset (Version 1.32.0). Zenodo. https://doi.org/10.5281/zenodo.21964209")

try:
    R = analyze(xml_bytes)
except Exception as e:                                   # malformed or non-bill XML
    st.error(f"Couldn't analyze {source}: {type(e).__name__}: {e}")
    st.stop()

meta, rules, tags, similar = R["meta"], R["rules"], R["tags"], R["similar"]
statements = rules[rules["item"] != "component"]
predicted = tags[tags["status"] == "predicted"]

# ---------------------------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------------------------
st.title(meta["short-title"] or meta["legis-num"] or source)
st.caption(" · ".join(x for x in [meta["legis-num"], meta["congress"], meta["sponsor"] and f"Sponsor: {meta['sponsor']}",
                                  meta["action-date"] and f"Introduced {meta['action-date']}"] if x))
if meta["official-title"]:
    st.write(meta["official-title"])

c = st.columns(5)
c[0].metric("Sections", len(R["segs"]))
c[1].metric("Provisions", len(R["prov"]))
c[2].metric("Duties", int(statements["hohfeld"].isin(["duty", "duty (not to)"]).sum()))
c[3].metric("AGORA tags predicted", len(predicted))
c[4].metric("Closest AGORA match", f"{similar['coverage'].iloc[0]:.2f}", help="Semantic coverage, 0–1")

tab_over, tab_prov, tab_law, tab_sim, tab_tags, tab_phr = st.tabs(
    ["Overview", "Provisions", "Legal effect", "Similar laws", "AGORA tags", "Phrases"])

# ---------------------------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------------------------
with tab_over:
    if R["prior"]:
        p = R["prior"]
        d = idx["docs"].loc[p["id"]]
        st.subheader("Likely earlier version found")
        st.info(f"**{d['Official name']}** ({d['Authority']}, {str(d['Most recent activity']).lower()}, proposed "
                f"{d['Proposed date']}) has lexical similarity **{p['lexical']:.2f}** to this bill, "
                f"{p['lexical'] / max(p['runner_up'], 1e-9):.1f}× the next closest AGORA document "
                f"({p['runner_up']:.2f}). [Source]({d['Link to document']})")
        show_changes(p["compare"], max_rows=5)
        st.caption("Full comparison in the **Similar laws** tab.")

    left, right = st.columns(2)
    with left:
        st.subheader("What the bill does")
        h = statements["hohfeld"].fillna("(other)").value_counts()
        st.write(", ".join(f"**{n}** {k}" for k, n in h.items()) or "No obligations or permissions detected.")
        actors = (statements["actor"].dropna().str.replace(r"^(?:the|an?|each|any)\s+", "", case=False, regex=True)
                  .str.strip().value_counts().head(5))
        if len(actors):
            st.write("Main actors: " + "; ".join(f"{a} ({n})" for a, n in actors.items()))
        inserted = rules["citation"].str.contains("→").mean() if len(rules) else 0
        if inserted > 0.5:
            st.write(f"**{inserted:.0%}** of the rules sit inside text this bill inserts into other laws, "
                     "so it works mainly by amending existing statutes.")
        top_acts = R["acts"].head(3)
        if len(top_acts):
            st.write("Most-cited statutes: " + "; ".join(top_acts.index))
    with right:
        st.subheader("AGORA taxonomy (predicted)")
        for dom in az.DOMAINS.values():
            hits = predicted.loc[predicted["domain"] == dom, "tag"].tolist()
            st.markdown(f"**{dom}:** " + ("; ".join(hits) if hits else "_none predicted_"))

    st.subheader("Drafting checks")
    if R["gaps"].empty:
        st.success("No gaps in provision numbering.")
    else:
        st.warning(f"{len(R['gaps'])} numbering gap(s): an enumerator is skipped.")
        st.dataframe(R["gaps"], hide_index=True)

    with st.expander("All metadata"):
        st.table(pd.Series(meta, name="value").dropna())

# ---------------------------------------------------------------------------------------------
# Provisions
# ---------------------------------------------------------------------------------------------
with tab_prov:
    st.caption("Text with a blue bar is inserted into another law (inside a quoted block).")
    prov = R["prov"]
    top = prov.index[(prov["depth"] == 0) & (prov["level"] == "section") & ~prov["quoted"]].tolist() + [len(prov)]
    for a, b in zip(top, top[1:]):
        r = prov.loc[a]
        with st.expander(f"{r['enum']} {r['header']}", expanded=False):
            st.markdown(outline_html(prov.loc[a:b - 1]), unsafe_allow_html=True)

    st.subheader(f"Definitions ({len(R['defs'])})")
    st.dataframe(R["defs"], hide_index=True, width="stretch",
                 column_config={"definition": st.column_config.TextColumn(width="large")})
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader(f"Cross-references ({len(R['refs'])})")
        st.dataframe(R["refs"], hide_index=True, width="stretch")
    with c2:
        st.subheader("Named statutes")
        st.dataframe(R["acts"], width="stretch")

# ---------------------------------------------------------------------------------------------
# Legal effect
# ---------------------------------------------------------------------------------------------
with tab_law:
    if rules.empty:
        st.info("No obligations, permissions or prohibitions detected.")
    else:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Hohfeldian positions")
            st.bar_chart(statements["hohfeld"].fillna("(other)").value_counts(), horizontal=True)
            st.caption("Counts statements and actionable list items; list components are excluded.")
        with c2:
            st.subheader("Qualifier rates")
            q = statements.groupby("type")[list(az.QUALIFIERS)].mean()
            st.dataframe(q.style.format("{:.0%}").background_gradient(cmap="Blues", vmin=0, vmax=1),
                         width="stretch")
            st.caption("Share of rules of each type containing a condition, exception, deadline, "
                       "hedge (“as appropriate”) or consultation requirement.")

        st.subheader("Rules")
        f1, f2 = st.columns([2, 1])
        types = f1.multiselect("Types", sorted(rules["type"].unique()), default=sorted(rules["type"].unique()))
        hide_components = f2.toggle("Hide list components", value=True)
        view = rules[rules["type"].isin(types)]
        if hide_components:
            view = view[view["item"] != "component"]
        st.dataframe(view[["citation", "type", "hohfeld", "actor", "action", "trigger"] + list(az.QUALIFIERS) + ["sentence"]],
                     hide_index=True, width="stretch",
                     column_config={"sentence": st.column_config.TextColumn(width="large")})

# ---------------------------------------------------------------------------------------------
# Similar laws
# ---------------------------------------------------------------------------------------------
with tab_sim:
    st.caption("**Semantic coverage**: average over the bill's substantive sections of the best meaning-based "
               "match in each document. **Best**: the single strongest section match. **Lexical**: shared "
               "wording across the whole document (TF-IDF).")
    n = st.slider("Documents shown", 5, 50, 15)
    show = similar.head(n)[["Casual name", "Authority", "Most recent activity", "coverage", "best",
                            "best section", "lexical", "Link to document"]]
    st.dataframe(show, width="stretch",
                 column_config={"_index": st.column_config.NumberColumn("AGORA ID", format="%d", width="small"),
                                "Casual name": st.column_config.TextColumn("Document", width="medium"),
                                "Authority": st.column_config.TextColumn(width="small"),
                                "Most recent activity": st.column_config.TextColumn("Status", width="small"),
                                "best section": st.column_config.TextColumn("Best section", width="small"),
                                "coverage": st.column_config.ProgressColumn("Semantic coverage", min_value=0, max_value=1, format="%.2f"),
                                "best": st.column_config.NumberColumn("Best", format="%.2f"),
                                "lexical": st.column_config.NumberColumn("Lexical", format="%.2f"),
                                "Link to document": LINK})

    options = similar.head(n).index.tolist()
    pick = st.selectbox("Look closer at", options, format_func=lambda i: f"{similar.loc[i, 'Casual name']} (AGORA {i})")
    d = similar.loc[pick]
    st.markdown(f"**{d['Official name']}**  \n{d['Authority']} · {d['Most recent activity']} · proposed {d['Proposed date']}")
    if isinstance(d["Short summary"], str):
        st.caption(d["Short summary"])

    st.markdown("##### Best-matching passage for each section")
    bp = passages(R["substantive"], pick)
    st.dataframe(bp, hide_index=True, width="stretch",
                 column_config={"similarity": st.column_config.NumberColumn(format="%.2f"),
                                "passage": st.column_config.TextColumn(width="large")})

    st.markdown("##### Compare as versions")
    words = len(doc_text(pick).split())
    if words > 30_000:
        st.caption(f"This document is {words:,} words; too long for a word-level comparison.")
    elif st.toggle("Show what changed between this document and the bill", value=R["prior"] is not None and R["prior"]["id"] == pick):
        show_changes(compare(R["bill_text"], pick))

    st.divider()
    st.subheader("Search AGORA by meaning")
    q = st.text_input("Describe a provision", placeholder="e.g. developers must label AI-generated images")
    if q:
        res = search(q)
        st.dataframe(res, hide_index=True, width="stretch",
                     column_config={"score": st.column_config.NumberColumn(format="%.2f"),
                                    "text": st.column_config.TextColumn("passage", width="large")})

# ---------------------------------------------------------------------------------------------
# AGORA tags
# ---------------------------------------------------------------------------------------------
with tab_tags:
    st.caption("First-pass predictions of the tags an AGORA annotator would apply, from an n-gram logistic "
               "regression trained on AGORA's human-annotated segments. **Evidence** lists the phrases that "
               "pushed hardest toward the tag. **Model F1** is how reliably that tag is predicted on documents "
               "the model hasn't seen (cross-validated); treat low-F1 or low-probability tags with caution.")
    tag_cols = {"probability": st.column_config.ProgressColumn("Probability", min_value=0, max_value=1, format="%.2f"),
                "threshold": st.column_config.NumberColumn("Threshold", format="%.2f"),
                "model F1 (cross-val)": st.column_config.NumberColumn("Model F1", format="%.2f")}
    st.subheader(f"Predicted ({len(predicted)})")
    st.dataframe(predicted.drop(columns="status"), hide_index=True, width="stretch", column_config=tag_cols)
    border = tags[tags["status"] == "borderline"]
    if len(border):
        st.subheader(f"Borderline ({len(border)})")
        st.caption("Reached at least 80% of the threshold without crossing it: worth a human look.")
        st.dataframe(border.drop(columns="status"), hide_index=True, width="stretch", column_config=tag_cols)

    if len(predicted):
        st.subheader("Where each predicted tag comes from")
        t = idx["tagger"]
        cols = [tg for tg in t["tags"] if az.short(tg) in set(predicted["tag"])]
        thr = pd.Series(t["thresholds"], index=t["tags"])[cols]
        ratio = (R["probs"][cols] / thr).clip(upper=2)
        fig, ax = plt.subplots(figsize=(max(6, 1.5 + 0.45 * len(cols)), 1.5 + 0.35 * len(ratio)))
        im = ax.imshow(ratio.to_numpy(), cmap="Blues", vmin=0, vmax=2, aspect="auto")
        ax.set_xticks(range(len(cols)), [az.short(tg) for tg in cols], rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(ratio)), ratio.index, fontsize=8)
        fig.colorbar(im, ax=ax, label="prob. ÷ threshold\n(≥ 1 means predicted)", shrink=0.8)
        plt.tight_layout()
        st.pyplot(fig, width="content")
        plt.close(fig)

# ---------------------------------------------------------------------------------------------
# Phrases
# ---------------------------------------------------------------------------------------------
with tab_phr:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Most frequent")
        st.caption("Phrase counts in this bill.")
        st.dataframe(ngram_table(R["frequent"], "{:g}"), width="stretch", height=560)
    with c2:
        st.subheader("Most distinctive")
        st.caption("TF-IDF against the AGORA corpus: frequent here, rare in other AI laws.")
        st.dataframe(ngram_table(R["distinctive"], "{:.3f}"), width="stretch", height=560)
