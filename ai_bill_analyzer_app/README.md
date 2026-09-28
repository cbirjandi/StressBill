# AI Bill Analyzer

A dashboard that analyzes a U.S. bill (XML from congress.gov or govinfo) against the AGORA dataset of AI
laws: its text and definitions, cross-references with links, obligations and permissions, the most similar
AI laws (including earlier versions of the same bill), predicted AGORA tags, and key phrases.

## Run it on your Mac

Double-click `start.command` (if macOS blocks it: System Settings → Privacy & Security → **Open Anyway**).
The first time it installs what it needs (a few minutes); the first start of the dashboard then builds its
AGORA index (about a minute). After that it opens in a few seconds.

## Put it online (Streamlit Community Cloud, free)

1. The repo needs `agora_dataset/` and `sample_bill/` (at the top level or next to `app.py`) plus the files
   in this folder.
2. Go to https://share.streamlit.io, sign in with GitHub, and click **Create app**.
3. Repository `cbirjandi/StressBill`, branch `main`, main file path `ai_bill_analyzer_app/app.py`.
   Under **Advanced settings**, choose Python 3.12. Click **Deploy**.
4. Wait for the install (a few minutes), then open the app once yourself: the first visit builds the index
   (about a minute). Then share the link.

Free apps go to sleep after a period without visitors; the next visitor wakes it and waits for the
one-minute rebuild.

## Files

| | |
|---|---|
| `app.py` | the dashboard |
| `analyzer.py` | the analysis: parsing, definitions, references, rules, similarity, tags, phrases |
| `build_artifacts.py` | builds the AGORA index; the app runs it automatically when needed |
| `requirements.txt` | Python packages |
| `start.command` | Mac launcher |

Data: Arnold, Z., Melot, J., Enwereazu, O., Schiff, D. S., Schiff, K. J., & Girard, T. (2026). AGORA Dataset
(Version 1.32.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.21964209
