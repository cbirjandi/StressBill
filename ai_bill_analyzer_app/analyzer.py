"""Bill analysis functions shared by build_artifacts.py (one-time training) and app.py (the dashboard).

Everything here is the notebook's logic, reorganized into functions that take their inputs
explicitly instead of relying on notebook globals.
"""
import difflib
import re

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# ---------------------------------------------------------------------------------------------
# 1. Parsing the bill XML (notebook sections 1-7)
# ---------------------------------------------------------------------------------------------
LEVELS = ["section", "subsection", "paragraph", "subparagraph",
          "clause", "subclause", "item", "subitem"]


def load_soup(xml):
    """Parse bill XML given as bytes, str, or a file path."""
    if isinstance(xml, (bytes, bytearray)):
        return BeautifulSoup(xml, "xml")
    if isinstance(xml, str) and xml.lstrip().startswith("<"):
        return BeautifulSoup(xml, "xml")
    with open(xml, encoding="utf-8") as f:
        return BeautifulSoup(f, "xml")


def _clean(el):
    """Text of an element with quoted words in quotation marks and tidy spacing."""
    if el is None:
        return ""
    el = BeautifulSoup(str(el), "xml")          # copy, so the original soup is untouched
    for q in el.find_all("quote"):
        q.insert(0, "“")
        q.append("”")
    return " ".join(el.get_text().split())


def _own(el, tag):
    return _clean(el.find(tag, recursive=False))


def _walk(el, path, depth, in_quote, rows):
    for child in el.find_all(LEVELS + ["quoted-block"], recursive=False):
        if child.name == "quoted-block":
            _walk(child, path + ["→"], depth, True, rows)       # "→": text inserted into another law
            continue
        enum = _own(child, "enum")
        new_path = path + [enum]
        rows.append({"citation": " ".join(p for p in new_path if p), "level": child.name,
                     "depth": depth, "enum": enum, "header": _own(child, "header"),
                     "text": _own(child, "text"), "continuation": _own(child, "continuation-text"),
                     "quoted": in_quote})
        _walk(child, new_path, depth + 1, in_quote, rows)


def parse_provisions(soup):
    """One row per provision in the bill body, with its citation and parent citation."""
    body = soup.find("legis-body") or soup
    rows = []
    _walk(body, [], 0, False, rows)
    df = pd.DataFrame(rows, columns=["citation", "level", "depth", "enum", "header",
                                     "text", "continuation", "quoted"])
    df["parent"] = df["citation"].str.rsplit(" ", n=1).str[0].where(df["depth"] > 0, "")
    return df


META_FIELDS = ["legis-num", "official-title", "short-title", "congress", "session",
               "current-chamber", "sponsor", "cosponsor", "committee-name", "action-date", "action-desc"]


def metadata(soup):
    return {f: (soup.find(f).get_text(" ", strip=True) if soup.find(f) else None) for f in META_FIELDS}


def _roman(s):
    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}
    s = s.lower()
    total = 0
    for a, b in zip(s, s[1:] + " "):
        total += -vals[a] if vals.get(b, 0) > vals[a] else vals[a]
    return total


def _ordinal(enum, level):
    e = enum.strip("().")
    try:
        if level in ("section", "paragraph"):
            return int(re.match(r"\d+", e).group())
        if level in ("subsection", "subparagraph"):
            return ord(e.lower()) - 96 if len(e) == 1 else None
        if level in ("clause", "subclause"):
            return _roman(e)
        if level in ("item", "subitem"):
            return ord(e[0].lower()) - 96
    except (AttributeError, KeyError, TypeError):
        return None


def numbering_gaps(prov):
    """Siblings whose enumerators skip a number, e.g. clause (iv) followed by (vi)."""
    gaps = []
    for (parent, level), g in prov.groupby(["parent", "level"], sort=False):
        nums = [_ordinal(e, level) for e in g["enum"]]
        pairs = list(zip(g["enum"], nums))
        for (e1, n1), (e2, n2) in zip(pairs, pairs[1:]):
            if n1 is not None and n2 is not None and n2 != n1 + 1:
                gaps.append({"parent": parent or "(bill)", "level": level, "from": e1, "to": e2})
    return pd.DataFrame(gaps, columns=["parent", "level", "from", "to"])


DEF_PAT = r"\b[Tt]he term “([^”]+)” (means|has the meaning given)"
SCOPE_PAT = r"\bIn this (Act|title|subtitle|section|subsection|paragraph)\b"


def definitions(prov):
    text_by_cite = dict(zip(prov["citation"], prov["text"]))
    rows = []
    for _, r in prov.iterrows():
        for m in re.finditer(DEF_PAT, r["text"]):
            s = re.search(SCOPE_PAT, r["text"]) or re.search(SCOPE_PAT, text_by_cite.get(r["parent"], ""))
            rows.append({"citation": r["citation"], "term": m.group(1),
                         "kind": "defined here" if m.group(2) == "means" else "borrowed",
                         "scope": s.group(1) if s else None,
                         "inserted into other law": r["quoted"], "definition": r["text"]})
    return pd.DataFrame(rows, columns=["citation", "term", "kind", "scope", "inserted into other law", "definition"])


def references(soup):
    rows = [{"type": x.name, "doc": x.get("legal-doc"), "text": x.get_text(strip=True),
             "cite": x.get("parsable-cite") or x.get("idref")}
            for x in soup.find_all(["external-xref", "internal-xref"])]
    return pd.DataFrame(rows, columns=["type", "doc", "text", "cite"])


ACT_PAT = (r"(?<=\bthe )“?((?:[A-Z][\w'.-]*|\([A-Z]\w*\))(?:,?\s+(?:(?:of|and|for|the|in|on|to)\s+)*"
           r"(?:[A-Z][\w'.-]*|\([A-Z]\w*\)))*\s+Act(?:\s+(?:of|for Fiscal Year)\s+\d{4}|,\s+\d{4})?)")


def named_acts(prov):
    found = prov["text"].str.extractall(ACT_PAT)
    if found.empty:
        return pd.Series(dtype=int, name="mentions").rename_axis("statute")
    return found[0].value_counts().rename("mentions").rename_axis("statute")


# ---------------------------------------------------------------------------------------------
# 2. Obligations, permissions, prohibitions (notebook section 8)
# ---------------------------------------------------------------------------------------------
PATTERNS = [
    ("construction",  r"\bnothing in this .{0,60}?shall be construed\b"),
    ("immunity",      r"\b(?:shall not be (?:liable|subject to)|no (?:cause of action|civil action) shall|(?:is|are|shall be) immune from)\b"),
    ("right",         r"\b(?:(?:is|are|shall be) entitled to|shall have the right to|may bring an? (?:civil )?action)\b"),
    ("appropriation", r"\b(?:is|are) (?:hereby )?(?:authorized to be )?appropriated\b"),
    ("prohibition",   r"\b(?:shall not|may not|must not|(?:is|are) prohibited from)\b"),
    ("obligation",    r"\b(?:shall|must|(?:is|are) required to)\b"),
    ("permission",    r"\b(?:may|(?:is|are) authorized to)\b"),
]
FALSE_ALARMS = (r"shall (?:also )?(?:mean|include|have the meaning)|may (?:also )?include|as may be \w+|"
                r"may be necessary|may be cited|as the case may be")
DEFINITION = r"\b(?:the|such) terms?\b.*\b(?:means|has the meaning|shall (?:also )?include)"
QUALIFIERS = {
    "condition": r"\b(?:if|unless|in the case of|upon|provided that|to be eligible|as a condition of)\b",
    "exception": r"\b(?:except|does not apply|shall not apply|exempt|notwithstanding)\b",
    "deadline":  r"\b(?:not later than|within \d+ (?:days|months|years)|annually|beginning on)\b",
    "hedge":     r"\b(?:as (?:\w+ and )?(?:appropriate|practicable)|to the extent (?:\w+ and )?practicable|as the .{1,80}? determines|as determined (?:appropriate )?by|reasonabl[ey]|seek to|endeavor to)\b",
    "consult":   r"\bin (?:consultation|coordination) with\b",
}
LIST_LEAD = r"(?:\bfollowing\b[^.]*:|—)\s*$"
POWER_VERBS = {"issue", "promulgate", "prescribe", "establish", "designate", "delegate", "waive",
               "impose", "enforce", "approve", "certify", "revoke", "suspend", "terminate",
               "exempt", "award", "assess", "order", "determine"}
POWER_OBJECTS = {"award", "grant", "rule", "regulation", "determination", "designation"}
HOHFELD = {"obligation": "duty", "prohibition": "duty (not to)", "right": "claim-right",
           "immunity": "immunity", "construction": None, "appropriation": None}


def _flag_qualifiers(text):
    return {q: bool(re.search(p, text, flags=re.I)) for q, p in QUALIFIERS.items()}


def _analyze_sentence(sent):
    s = sent.text
    if re.search(DEFINITION, s, flags=re.I):
        return None
    masked = re.sub(FALSE_ALARMS, lambda m: " " * len(m.group()), s, flags=re.I)
    for kind, pat in PATTERNS:
        m = re.search(pat, masked, flags=re.I)
        if m:
            break
    else:
        return None

    tok = next((t for t in sent if t.idx == sent.start_char + m.start()), None)
    actor = action = None
    objects = set()
    if tok is not None:
        head = tok.head
        subj = [c for c in head.children if c.dep_ in ("nsubj", "nsubjpass")]
        if subj:
            actor = sent.doc[subj[0].left_edge.i: subj[0].right_edge.i + 1].text.split(",")[0].strip()
        # Passive phrasing like "is required to conduct": the real action is the complement verb
        if tok.lower_ in {"is", "are"} and head.lemma_ in {"require", "authorize", "prohibit", "entitle"}:
            xcomp = [c for c in head.children if c.dep_ == "xcomp"]
            head = xcomp[0] if xcomp else head
        # (The notebook computed action/objects only when no subject was found; fixed here.)
        action = head.lemma_
        objects = {c.lemma_.lower() for c in head.children if c.dep_ == "dobj"}
    if actor is None:
        before = s[:m.start()].split(",")[-1]
        actor = re.split(r"\s+(?:that|which|who)\s+", before)[0].strip() or None

    hohfeld = HOHFELD.get(kind)
    if kind == "permission":
        hohfeld = "power" if action in POWER_VERBS or objects & POWER_OBJECTS else "privilege"
    row = {"type": kind, "hohfeld": hohfeld, "actor": actor, "action": action,
           "trigger": m.group(0), "sentence": s, "inherited": False, "item": "statement"}
    row.update(_flag_qualifiers(s))
    return row


def deontic_rules(prov, nlp):
    """Obligations, permissions, etc., one row per rule, with list items inheriting their lead-in."""
    rows, leads = [], {}
    for cite, parent, text in zip(prov["citation"], prov["parent"], prov["text"].fillna("")):
        doc = nlp(text)
        found = [r for r in (_analyze_sentence(s) for s in doc.sents) if r]
        if not found and parent in leads and text:
            base = leads[parent]
            starts_with_verb = doc[0].tag_ == "VB"
            r = {**base, "action": doc[0].lemma_.lower() if starts_with_verb else base["action"],
                 "trigger": "(inherited)", "sentence": text, "inherited": True,
                 "item": "action" if starts_with_verb else "component"}
            r.update(_flag_qualifiers(text))
            found = [r]
        for r in found:
            rows.append({"citation": cite, **r})
        if found and re.search(LIST_LEAD, text):
            leads[cite] = found[-1]
    cols = ["citation", "type", "hohfeld", "actor", "action", "trigger", "sentence",
            "inherited", "item"] + list(QUALIFIERS)
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------------------------
# 3. AGORA-style segments and n-grams (notebook sections 10-11)
# ---------------------------------------------------------------------------------------------
def _provision_text(r):
    head = f"{r['header']}.--" if r["header"] else ""
    return " ".join(x for x in [r["enum"], head + r["text"], r["continuation"]] if x)


def bill_segments(prov):
    """One AGORA-style segment per top-level section."""
    rows = []
    for _, r in prov.iterrows():
        if r["level"] == "section" and r["depth"] == 0 and not r["quoted"]:
            first = f"SEC. {r['enum']} {r['header'].upper()}." + (f" {r['text']}" if r["text"] else "")
            rows.append({"citation": f"Sec. {r['enum'].rstrip('.')}", "header": r["header"],
                         "parts": [first] + ([r["continuation"]] if r["continuation"] else [])})
        elif rows:
            rows[-1]["parts"].append(_provision_text(r))
    if not rows:
        rows = [{"citation": "Whole bill", "header": "", "parts": [_provision_text(r) for _, r in prov.iterrows()]}]
    out = pd.DataFrame(rows)
    out["text"] = out.pop("parts").apply(lambda p: " ".join(x for x in p if x))
    out["words"] = out["text"].str.split().str.len()
    return out


LEGAL_STOP = {"shall", "sec", "section", "sections", "subsection", "paragraph", "subparagraph", "clause",
              "subclause", "act", "term", "means", "include", "includes", "including", "pursuant",
              "described", "thereof", "herein", "such", "may", "applicable", "respect", "purposes", "et", "seq"}
STOP = ENGLISH_STOP_WORDS | LEGAL_STOP
TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z-]+\b"
_WORD = re.compile(r"[a-z][a-z-]*[a-z]")
_BREAK = re.compile(r"[.;:!?()\[\]“”\"—]|--|\n\n")


def phrases(text, n_max=3):
    """All 1- to n_max-word phrases that don't start or end with a stop word or cross punctuation."""
    out = []
    for piece in _BREAK.split(text.lower()):
        toks = _WORD.findall(piece)
        for n in range(1, n_max + 1):
            for i in range(len(toks) - n + 1):
                if toks[i] not in STOP and toks[i + n - 1] not in STOP:
                    out.append(" ".join(toks[i:i + n]))
    return out


def top_ngrams(text, n, k=15):
    cv = CountVectorizer(analyzer=lambda t: [p for p in phrases(t) if p.count(" ") == n - 1])
    try:
        X = cv.fit_transform([text])
    except ValueError:                              # no phrases of this length
        return pd.Series(dtype=float)
    return pd.Series(np.asarray(X.sum(axis=0)).ravel(), cv.get_feature_names_out()).nlargest(k)


def distinctive_ngrams(text, doc_vec, k=15):
    """Top TF-IDF phrases of `text` against the AGORA corpus, by phrase length."""
    w = pd.Series(doc_vec.transform([text]).toarray().ravel(), doc_vec.get_feature_names_out())
    w = w[w > 0].sort_values(ascending=False)
    n_words = w.index.str.count(" ") + 1
    return {n: w[n_words == n].head(k) for n in (1, 2, 3)}


# ---------------------------------------------------------------------------------------------
# 4. Embeddings and similarity (notebook section 12)
# ---------------------------------------------------------------------------------------------
BOILERPLATE = (r"short title|table of contents|severability|effective date|authorization of appropriations|"
               r"rule of construction|sense of congress|findings")


def chunk_words(text, size=150, overlap=30):
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


def make_chunks(seg_text):
    """Chunk every AGORA segment. `seg_text` must be sorted by document, then position."""
    return pd.DataFrame([(d, p, c) for d, p, t in zip(seg_text["doc_id"], seg_text["position"], seg_text["text"])
                         for c in chunk_words(t)], columns=["doc_id", "position", "text"])


class LSAEmbedder:
    """Light, torch-free stand-in for a sentence-transformer: TF-IDF compressed with SVD."""

    def __init__(self, tfidf, components):
        self.tfidf, self.components = tfidf, components.astype(np.float32)

    def __call__(self, texts):
        Z = self.tfidf.transform(list(texts)) @ self.components.T
        return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def sentence_transformer_embedder(model_name):
    from sentence_transformers import SentenceTransformer        # imported only when needed
    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=len(texts) > 500)


def section_by_document(texts, embed, E, doc_starts):
    """Matrix (sections x AGORA documents): best chunk-to-chunk cosine similarity."""
    q_chunks = [chunk_words(t) for t in texts]
    Q = embed([c for cs in q_chunks for c in cs]).astype(np.float32)
    q_starts = np.cumsum([0] + [len(cs) for cs in q_chunks[:-1]])
    S = Q @ E.T
    S = np.maximum.reduceat(S, doc_starts, axis=1)
    return np.maximum.reduceat(S, q_starts, axis=0)


def best_passages(seg_df, doc_id, embed, E, chunks):
    """For each bill section, the most similar chunk of one AGORA document."""
    mask = (chunks["doc_id"] == doc_id).to_numpy()
    doc_chunks = chunks.loc[mask, "text"].tolist()
    rows = []
    for _, s in seg_df.iterrows():
        S = embed(chunk_words(s["text"])).astype(np.float32) @ E[mask].T
        rows.append({"section": f"{s['citation']} {s['header']}".strip(), "similarity": float(S.max()),
                     "passage": doc_chunks[int(np.unravel_index(S.argmax(), S.shape)[1])]})
    return pd.DataFrame(rows)


def compare_versions(new_text, old_text, min_words=12):
    """Word-level comparison of two versions of a bill: overlap shares and the larger changes."""
    split = lambda t: re.sub(r"\.?(?:--|—|–)", ". ", t).split()      # "Header.--Text" -> "Header. Text"
    key = lambda w: re.sub(r"[^\w]", "", w.lower())                   # compare ignoring case and punctuation
    a, b = split(old_text), split(new_text)
    sm = difflib.SequenceMatcher(None, [key(w) for w in a], [key(w) for w in b], autojunk=False)
    same = sum(m.size for m in sm.get_matching_blocks())
    changes = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal" and max(i2 - i1, j2 - j1) >= min_words:
            changes.append({"change": {"insert": "added", "delete": "removed"}.get(op, "rewritten"),
                            "words": max(i2 - i1, j2 - j1),
                            "removed text": " ".join(a[i1:i2]), "added text": " ".join(b[j1:j2])})
    return {"old words": len(a), "new words": len(b),
            "share of old kept": same / max(len(a), 1), "share of new that is old": same / max(len(b), 1),
            "changes": pd.DataFrame(changes, columns=["change", "words", "removed text", "added text"])
                         .sort_values("words", ascending=False).reset_index(drop=True)}


# ---------------------------------------------------------------------------------------------
# 5. AGORA tags (notebook sections 9 and 13)
# ---------------------------------------------------------------------------------------------
DOMAINS = {"Risk factors": "Risk factors governed", "Harms": "Harms addressed",
           "Strategies": "Governance strategies", "Incentives": "Incentives for compliance",
           "Applications": "Application domains addressed"}


def domain(tag):
    return DOMAINS[tag.split(": ")[0]]


def short(tag):
    return tag.split(": ", 1)[1]


def parent_map(tags):
    return {t: ": ".join(t.split(": ")[:2]) for t in tags
            if t.count(": ") == 2 and ": ".join(t.split(": ")[:2]) in tags}


def add_parents(Y):
    """Switch on each parent tag wherever one of its subtags is on (DataFrame of 0/1 or bool)."""
    Y = Y.copy()
    for child, parent in parent_map(list(Y.columns)).items():
        Y[parent] = Y[parent] | Y[child]
    return Y


def tag_segments(seg_df, tfidf, model, thresholds, tags, reliability=None, borderline=0.8, k_evidence=5):
    """Predict AGORA tags per section and roll them up to the whole bill.

    Returns (tag table, section x tag probability table)."""
    texts = seg_df["text"].tolist()
    X = tfidf.transform(texts)
    P = model.predict_proba(X)
    thr = np.asarray(thresholds)
    pred = add_parents(pd.DataFrame(P >= thr, columns=tags)).to_numpy()
    features = tfidf.get_feature_names_out()
    rows = []
    for j, tag in enumerate(tags):
        best_i = int(P[:, j].argmax())
        hit = pred[:, j].astype(bool)
        if not (hit.any() or P[best_i, j] >= borderline * thr[j]):
            continue
        coef = model.estimators_[j].coef_.ravel()
        contrib = X[best_i].toarray().ravel() * coef
        top = [features[i] for i in contrib.argsort()[::-1][:k_evidence] if contrib[i] > 0]
        rows.append({"domain": domain(tag), "tag": short(tag),
                     "status": "predicted" if hit.any() else "borderline",
                     "probability": round(float(P[best_i, j]), 2), "threshold": float(thr[j]),
                     "sections": ", ".join(seg_df["citation"].to_numpy()[hit]) or seg_df["citation"].iloc[best_i],
                     "evidence": ", ".join(top),
                     "model F1 (cross-val)": None if reliability is None else reliability.get(tag)})
    table = pd.DataFrame(rows, columns=["domain", "tag", "status", "probability", "threshold", "sections",
                                        "evidence", "model F1 (cross-val)"])
    order = {d: i for i, d in enumerate(DOMAINS.values())}
    table = table.sort_values(["status", "domain", "probability"], key=lambda c: c.map(order) if c.name == "domain" else c,
                              ascending=[False, True, False]).reset_index(drop=True)
    return table, pd.DataFrame(P, index=seg_df["citation"], columns=tags)
