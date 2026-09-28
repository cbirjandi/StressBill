"""AI Bill Analyzer: Streamlit dashboard.

    streamlit run app.py

On first start it builds its AGORA index (about a minute); after that it loads in seconds.
"""
import html
import os
import re
import threading

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st

import analyzer as az
import build_artifacts

HERE = os.path.dirname(os.path.abspath(__file__))
ART = build_artifacts.DEFAULT_OUT
SAMPLE = next((p for p in (os.path.join(d, "sample_bill", "BILLS-119hr9334ih (1).xml")
                           for d in (HERE, os.path.dirname(HERE))) if os.path.exists(p)), "")
PRIOR_VERSION_CUTOFF = 0.5          # lexical similarity above which a match is flagged as a likely earlier version

st.set_page_config(page_title="AI Bill Analyzer", page_icon="⚖️", layout="wide")


# ---------------------------------------------------------------------------------------------
# Index: built once per server, then loaded once per server process
# ---------------------------------------------------------------------------------------------
@st.cache_resource
def _build_lock():
    return threading.Lock()


def ensure_index():
    with _build_lock():                         # one build at a time, even with several visitors
        if not build_artifacts.is_current(ART):
            with st.status("First start: building the AGORA index (about a minute)…", expanded=True) as box:
                build_artifacts.build(out=ART, log=box.write)
                box.update(label="AGORA index built", state="complete", expanded=False)


@st.cache_resource(show_spinner="Loading the AGORA index…")
def load_index():
    doc_ids = np.load(f"{ART}/doc_ids.npy")
    return {"tagger": joblib.load(f"{ART}/tagger.joblib"),
            "docs": pd.read_parquet(f"{ART}/documents.parquet").loc[doc_ids],
            "seg_text": pd.read_parquet(f"{ART}/segments_text.parquet"),
            "doc_vec": joblib.load(f"{ART}/doc_vectorizer.joblib"),
            "X_docs": sp.load_npz(f"{ART}/doc_matrix.npz"),
            "X_segs": sp.load_npz(f"{ART}/segment_matrix.npz").tocsr()}


@st.cache_resource(show_spinner="Loading the language model…")
def load_nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def doc_segments(doc_id):
    s = load_index()["seg_text"]
    return s[s["doc_id"] == doc_id]


@st.cache_data(show_spinner="Analyzing the bill…", max_entries=10)
def analyze(xml_bytes):
    idx = load_index()
    soup = az.load_soup(xml_bytes)
    prov = az.parse_provisions(soup)
    segs = az.bill_segments(prov)
    bill_text = " ".join(segs["text"])

    sims = (idx["X_docs"] @ idx["doc_vec"].transform([bill_text]).T).toarray().ravel()
    similar = idx["docs"].assign(similarity=sims).sort_values("similarity", ascending=False)
    prior = None
    if similar["similarity"].iloc[0] >= PRIOR_VERSION_CUTOFF:
        pid = similar.index[0]
        prior = {"id": int(pid), "similarity": float(similar["similarity"].iloc[0]),
                 "runner_up": float(similar["similarity"].iloc[1]),
                 "compare": az.compare_versions(bill_text, " ".join(doc_segments(pid)["text"]))}

    t = idx["tagger"]
    tags, probs = az.tag_segments(segs, t["tfidf"], t["model"], t["thresholds"], t["tags"], t["reliability"])
    substantive = segs[~segs["header"].str.contains(az.BOILERPLATE, case=False, regex=True)]

    return {"meta": az.metadata(soup), "prov": prov, "segs": segs,
            "substantive": substantive if len(substantive) else segs, "bill_text": bill_text,
            "gaps": az.numbering_gaps(prov), "defs": az.definitions(prov), "refs": az.references(soup),
            "acts": az.named_acts(prov), "rules": az.deontic_rules(prov, load_nlp()),
            "similar": similar, "prior": prior, "tags": tags, "probs": probs,
            "frequent": {n: az.top_ngrams(bill_text, n) for n in (1, 2, 3)},
            "distinctive": az.distinctive_ngrams(bill_text, idx["doc_vec"])}


@st.cache_data(show_spinner="Comparing texts…", max_entries=20)
def compare(bill_text, doc_id):
    return az.compare_versions(bill_text, " ".join(doc_segments(doc_id)["text"]))


# ---------------------------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------------------------
def md(text):
    """Escape characters Streamlit markdown would otherwise interpret ($ starts LaTeX, * italics, ...)."""
    return re.sub(r"([\\`*_\[\]$~#<>|])", r"\\\1", str(text))


def ngram_table(by_n, fmt):
    cols = {f"{n}-word": pd.Series([f"{p}  ({fmt.format(v)})" for p, v in s.items()]) for n, s in by_n.items()}
    out = pd.DataFrame(cols).fillna("")
    out.index += 1
    return out


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
# Sidebar and setup
# ---------------------------------------------------------------------------------------------
st.sidebar.title("⚖️ AI Bill Analyzer")
upload = st.sidebar.file_uploader("Upload a bill (XML from congress.gov or govinfo)", type=["xml"])
if upload is not None:
    xml_bytes, source = upload.getvalue(), upload.name
elif SAMPLE:
    with open(SAMPLE, "rb") as f:
        xml_bytes, source = f.read(), "Sample: H.R. 9334"
    st.sidebar.caption("No file uploaded, so showing the sample bill (H.R. 9334).")
else:
    st.info("Upload a bill's XML file in the sidebar to begin.")
    st.stop()

ensure_index()
idx = load_index()
st.sidebar.divider()
st.sidebar.caption(f"Compared against the AGORA dataset of {len(idx['docs']):,} AI laws and policies. "
                   f"Tags predicted by a model trained on {idx['tagger']['n_segments']:,} human-annotated segments.")
st.sidebar.caption("Data: Arnold, Melot, Enwereazu, Schiff, Schiff & Girard (2026), AGORA Dataset v1.32.0, "
                   "[doi:10.5281/zenodo.21964209](https://doi.org/10.5281/zenodo.21964209)")

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
c[4].metric("Closest AGORA match", f"{similar['similarity'].iloc[0]:.2f}",
            help="Share of distinctive wording in common with the most similar AGORA document (TF-IDF cosine, 0–1)")

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
        st.info(f"**{md(d['Official name'])}** ({d['Authority']}, {str(d['Most recent activity']).lower()}, proposed "
                f"{d['Proposed date']}) shares much of this bill's wording: similarity **{p['similarity']:.2f}**, "
                f"{p['similarity'] / max(p['runner_up'], 1e-9):.1f}× the next closest AGORA document "
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
            st.write("Main actors: " + "; ".join(f"{md(a)} ({n})" for a, n in actors.items()))
        inserted = rules["citation"].str.contains("→").mean() if len(rules) else 0
        if inserted > 0.5:
            st.write(f"**{inserted:.0%}** of the rules sit inside text this bill inserts into other laws, "
                     "so it works mainly by amending existing statutes.")
        if len(R["acts"]):
            st.write("Most-cited statutes: " + "; ".join(md(a) for a in R["acts"].head(3).index))
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
    st.subheader("Text")
    st.caption("Text with a blue bar is inserted into another law (inside a quoted block).")
    prov = R["prov"]
    top = prov.index[(prov["depth"] == 0) & (prov["level"] == "section") & ~prov["quoted"]].tolist() + [len(prov)]
    for a, b in zip(top, top[1:]):
        r = prov.loc[a]
        with st.expander(f"{r['enum']} {r['header']}"):
            st.markdown(outline_html(prov.loc[a:b - 1]), unsafe_allow_html=True)

    defs = R["defs"]
    st.subheader(f"Definitions ({len(defs)})")
    if defs.empty:
        st.caption("No defined terms found.")
    else:
        st.caption(" · ".join(md(t) for t in defs["term"]))
        i = st.selectbox("Read a definition in full", defs.index, format_func=lambda k: defs.loc[k, "term"])
        d = defs.loc[i]
        notes = ["defined in this bill" if d["kind"] == "defined here" else "borrowed from another law"]
        if isinstance(d["scope"], str):
            notes.append(f"applies to this {d['scope'].lower()}")
        if d["inserted into other law"]:
            notes.append("inserted into another law")
        with st.container(border=True):
            st.markdown(f"**{md(d['term'])}**")
            st.caption(f"{d['citation']} · " + "; ".join(notes))
            st.write(md(d["definition"]))

    refs = R["refs"]
    st.subheader(f"Cross-references ({len(refs)})")
    if refs.empty:
        st.caption("No cross-references to other laws found.")
    else:
        st.dataframe(refs, hide_index=True, width="stretch",
                     column_config={"reference": st.column_config.TextColumn("Reference", width="medium"),
                                    "kind": "Type", "mentions": st.column_config.NumberColumn("Mentions", width="small"),
                                    "source": "Source",
                                    "link": st.column_config.LinkColumn("Read it", display_text="open ↗")})
    if len(R["acts"]):
        st.subheader("Statutes named in the text")
        st.dataframe(R["acts"].reset_index(), hide_index=True,
                     column_config={"statute": "Statute", "mentions": "Mentions"})

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
            st.dataframe(q.style.format("{:.0%}").background_gradient(cmap="Blues", vmin=0, vmax=1), width="stretch")
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
    st.caption("AGORA documents ranked by how much distinctive wording they share with this bill "
               "(TF-IDF cosine similarity: 1 = identical wording, 0 = nothing in common).")
    n = st.slider("Documents shown", 5, 50, 15)
    top_docs = similar.head(n)
    st.dataframe(top_docs[["Casual name", "Authority", "Most recent activity", "Proposed date", "similarity", "Link to document"]]
                 .fillna({"Proposed date": "—"}),
                 width="stretch",
                 column_config={"_index": st.column_config.NumberColumn("AGORA ID", format="%d", width="small"),
                                "Casual name": st.column_config.TextColumn("Document", width="medium"),
                                "Authority": st.column_config.TextColumn(width="small"),
                                "Most recent activity": st.column_config.TextColumn("Status", width="small"),
                                "Proposed date": st.column_config.TextColumn("Proposed", width="small"),
                                "similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.2f"),
                                "Link to document": st.column_config.LinkColumn("Source", display_text="open ↗")})

    pick = st.selectbox("Look closer at", top_docs.index,
                        format_func=lambda i: f"{similar.loc[i, 'Casual name']} (AGORA {i})")
    d = similar.loc[pick]
    with st.container(border=True):
        st.markdown(f"**{md(d['Official name'])}**  \n{d['Authority']} · {d['Most recent activity']} · proposed {d['Proposed date'] or '—'}")
        if isinstance(d["Short summary"], str):
            st.caption(d["Short summary"])

    st.markdown("##### Closest passage for each section")
    cp = az.closest_passages(R["substantive"], doc_segments(pick), idx["doc_vec"])
    st.dataframe(cp, hide_index=True, width="stretch",
                 column_config={"section": st.column_config.TextColumn("Bill section", width="medium"),
                                "similarity": st.column_config.NumberColumn("Similarity", format="%.2f", width="small"),
                                "passage": st.column_config.TextColumn("Closest passage in the AGORA document", width="large")})

    st.markdown("##### Compare as versions")
    words = sum(len(t.split()) for t in doc_segments(pick)["text"])
    if words > 30_000:
        st.caption(f"This document is {words:,} words; too long for a word-level comparison.")
    elif st.toggle("Show what changed between this document and the bill",
                   value=R["prior"] is not None and R["prior"]["id"] == pick):
        show_changes(compare(R["bill_text"], pick))

    st.divider()
    st.subheader("Search AGORA")
    q = st.text_input("Search every AGORA segment by keywords", placeholder="e.g. watermark AI-generated content")
    if q:
        res = az.keyword_search(q, idx["doc_vec"], idx["X_segs"], idx["seg_text"])
        if res.empty:
            st.caption("No matches. Try other words; very common and very rare words are not indexed.")
        else:
            res = res.assign(document=res["doc_id"].map(idx["docs"]["Casual name"]))
            st.dataframe(res[["score", "document", "doc_id", "text"]], hide_index=True, width="stretch",
                         column_config={"score": st.column_config.NumberColumn("Score", format="%.2f", width="small"),
                                        "doc_id": st.column_config.NumberColumn("AGORA ID", format="%d", width="small"),
                                        "text": st.column_config.TextColumn("Passage", width="large")})

# ---------------------------------------------------------------------------------------------
# AGORA tags
# ---------------------------------------------------------------------------------------------
with tab_tags:
    st.caption("First-pass predictions of the tags an AGORA annotator would apply, from an n-gram logistic "
               "regression trained on AGORA's human-annotated segments. **Evidence** lists the phrases that "
               "pushed hardest toward the tag. **Model F1** is how reliably that tag is predicted on documents "
               "the model hasn't seen (cross-validated); treat low-F1 or low-probability tags with caution.")
    tag_cols = {"domain": st.column_config.TextColumn("Domain", width="small"),
                "tag": st.column_config.TextColumn("Tag", width="medium"),
                "probability": st.column_config.ProgressColumn("Probability", min_value=0, max_value=1, format="%.2f"),
                "threshold": st.column_config.NumberColumn("Threshold", format="%.2f", width="small"),
                "sections": st.column_config.TextColumn("Sections", width="small"),
                "evidence": st.column_config.TextColumn("Evidence", width="medium"),
                "model F1 (cross-val)": st.column_config.NumberColumn("Model F1", format="%.2f", width="small")}
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
