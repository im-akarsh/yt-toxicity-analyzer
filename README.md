# YouTube Comment Toxicity Analyzer
Finding the toxicity of a YouTube channel's comment section and which videos drive it. The score is build using a toxicity classifier trained on the Jigsaw dataset along with a separate classifier for Hinglish. ALso, clustering to spot patterns across videos.

## Links
[Live Web App]

## Objective
A YouTube channel's comment section can be place for both encouraging and criticizing comments, but sometimes the limits were cross making the comment section toxic. The project aims to find a score of toxicity per video and whether certain topics or formats attract more toxic reactions than others.

## Tech Stack & Tools
- **Python Language**: `python 3.14`
- **Data Collection**: YouTube Data API v3, `google-api-python-client`
- **Data Preparation & Model Training**: `pandas`, `numpy`, `scikit-learn`, `langdetect`
- **Deployment**: `streamlit` for running and deploying the app.

## Repository Structure
```
yt-toxicity-analyzer/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_video_clustering.ipynb
├── models/
├── data/
├── resources/
├── app.py
├── requirements.txt
└── README.md
```

## Pipeline

### Data Preparation:
- **Notebook**: `01_data_preparation.ipynb`
- **Data Used**: Scraped comments and video metadata from a real Indian YouTube channel (@SundaySarthak), 12,521 comments across 50 videos for a baseline score.
- **Data Processing**:
  - Cleaned raw comment text (emoji, URLs, timestamps, HTML symbols).
  - Detected each comment's language.
    - ***The Hinglish Problem***: The comments section consist of Hinglish. There is no actual way to   translate or  `langdetect` not even detect properly assigning random language to it. No fixes I try handle these comments work properly. The translating to english not yield any good results and another method to translating to *devanagiri* and than to english as the comments quality is not reliable at all. These comments are flagged as hinglih comments.
    - The non-english, non-hinglish comments are translated to english.
  - **Target Feature**: multi-label toxicity (toxic, severe toxic, obscene, threat, insult, identity hate), trained separately on the Jigsaw dataset.

### Model Training:
- **Notebook**: `02_model_training.ipynb`
- Logistic Regression vs. Linear SVC compared on the Jigsaw dataset, SVC won on every metric, calibrated with `CalibratedClassifierCV` so it could still produce usable probabilities for per-label threshold tuning.
- A separate TF-IDF + Logistic Regression model trained on the Hinglish dataset (0.95 Macro F1 on validation).
- The scraped processed comments were trained with first with only Jigsaw model and than with routing the hinglish flagged comments to Hinglish model separately.
- Without the Hinglish routing, my pipeline found 323 toxic comments out of 12,521. With it, that jumped to 742 — the English-only version was missing a lot of real toxicity just because it couldn't understand the language it was written in.

### Video-Level Aggregation & Clustering:
- **Notebook**: `03_video_clustering.ipynb`
- Per video comment toxicity  — ranged from 1.8% to 18.7% across the 50 videos
- Clustered videos on toxicity rate, views, and likes. We find out, Angry, single-topic videos got more toxic comments than calm, multi-topic roundups — regardless of how many views either got.

### Deployment
- Streamlit app takes any YouTube channel handle, scans its recent videos and comments live, and returns a toxicity report using the same routed scoring (Jigsaw + Hinglish) as the notebooks.

## Key Decisions & Findings
- **Hinglish routing over translation** — a dedicated classifier for Hinglish comments, rather than trying to translate everything to English at scale.
- **Tested the app on Vsauce** (a calm science channel) as a control — it correctly showed low toxicity overall, but flagged one joke-heavy video too high showing the model not handling the sarcasm for now.
- **Clustering on numeric signals only** — dropped title-word features after they broke the clustering (one giant cluster, three tiny ones), since with only 50 videos, sparse text features overwhelmed the three meaningful numeric ones.

## Launching
```bash
pip install -r requirements.txt
streamlit run app.py
```
You'll need a YouTube Data API key (free from Google Cloud Console). Put it in a `.env` file:
```
YOUTUBE_API_KEY=your_key_here
```

## Limitation & Future Improvements
- Comments genuinely written in non-English scripts other than the ones tested here may still fall through to the English model if `langdetect` doesn't flag them correctly.
- The translation step for non-English comments works well when that bucket is small. For a channel where non-English content is the majority language, translating at that scale would likely hit the rate-limit issues.
- The classifier conflicts profanity with toxicity, since it's a lexical (word-presence) model rather than one that understands context — a transformer-based model would likely handle jokes vs. real hostility better.
- A better long-term fix for Hinglish specifically would be a multilingual model built for Indian languages (like MuRIL or XLM-RoBERTa) fine-tuned on code-mixed hate speech data, rather than a small dedicated classifier — this was out of scope here but is the clear next step.