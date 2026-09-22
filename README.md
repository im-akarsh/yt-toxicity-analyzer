# YouTube Comment Toxicity Analyzer
Scoring the toxicity of a YouTube channel's comment section, and finding which videos or topics drive it. Built using a toxicity classifier trained on the Jigsaw dataset, a separate classifier for Hinglish, and clustering to spot patterns across videos.

## Links
[Live Web App]

## Objective
A YouTube channel's comment section can range from calm to genuinely hostile, and it's not always obvious why. This project scores comments for toxicity, rolls that up to a per-video toxicity rate, and clusters videos to find out whether certain topics or formats attract more toxic reactions than others.

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
├── app.py
├── requirements.txt
└── README.md
```

## Pipeline

### Data Preparation:
- **Notebook**: `01_data_preparation.ipynb`
- **Data Used**: Scraped comments and video metadata from a real Indian YouTube channel (@SundaySarthak), 12,521 comments across 50 videos.
- **Data Processing**:
  - Cleaned raw comment text (emoji, URLs, timestamps, HTML symbols).
  - Detected each comment's language.
  - **Target Feature**: multi-label toxicity (toxic, severe toxic, obscene, threat, insult, identity hate), trained separately on the Jigsaw dataset.

### The Hinglish Problem
This is the part of the project I'm actually most proud of, even though it's not something that "worked perfectly."

The toxicity model is trained on Jigsaw, which is all English. But when I looked at the comments I scraped, a huge share of them (45.7%) were in Hinglish — Hindi written in English letters, like "yeh video bahut acha hai." `langdetect` could not handle this at all — it guessed random unrelated languages (Indonesian, Somali, Swahili, even Welsh) for these comments, since short romanized Hindi words happen to share letter patterns with them.

I tested a few fixes:
- Translating everything to English (Google Translate) — hit rate limits and took over an hour, not practical.
- Converting to Devanagari first, then translating — transliteration quality wasn't reliable, sometimes turning English words like "video" into gibberish.
- A couple of free/offline translation tools — installation issues, didn't get these working.

None of these worked well enough at this scale, so instead I trained a **second, separate model** just for Hinglish, using a small labeled dataset (1,366 comments) of Hinglish text marked offensive or not. Comments flagged as likely Hinglish get routed to this model instead of the English one.

Genuinely non-English comments (proper Hindi, Tamil, Chinese, etc. — a much smaller ~3.6% of the dataset) get translated to English before scoring, since that bucket is small enough for translation to actually work without hitting the same rate-limit wall.

**This made a real difference.** Without the Hinglish routing, my pipeline found 323 toxic comments out of 12,521. With it, that jumped to 742 — the English-only version was missing a lot of real toxicity just because it couldn't understand the language it was written in.

### Model Training:
- **Notebook**: `02_model_training.ipynb`
- Logistic Regression vs. Linear SVC compared on the Jigsaw dataset, SVC won on every metric, calibrated with `CalibratedClassifierCV` so it could still produce usable probabilities for per-label threshold tuning.
- A separate TF-IDF + Logistic Regression model trained on the Hinglish dataset (0.95 Macro F1 on validation).

### Video-Level Aggregation & Clustering:
- **Notebook**: `03_video_clustering.ipynb`
- Rolled comment-level toxicity up to a per-video toxicity rate — ranged from 1.8% to 18.7% across the 50 videos, even though nearly the entire channel (49 of 50 videos) falls under the same YouTube category.
- Clustered videos on toxicity rate, views, and likes (an earlier attempt using title-word features alongside these collapsed 45 of 50 videos into one cluster, since sparse word features drowned out the numeric signal — dropped for the numeric-only version, which produced 4 well-balanced clusters).
- **Finding**: single-issue, outrage-framed videos ("Shameful," "Why Are Reporters Targeted?") consistently had the highest toxicity, while multi-topic "Sunday Show"/"Midweek" roundup videos were consistently the calmest — even when the roundups got comparable or higher views. Toxicity here tracks content format, not popularity.

### Deployment
- Streamlit app takes any YouTube channel handle, scans its recent videos and comments live, and returns a toxicity report using the same routed scoring (Jigsaw + Hinglish) as the notebooks.

## Key Decisions & Findings
- **Hinglish routing over translation** — a dedicated classifier for Hinglish comments, rather than trying to translate everything to English at scale, since translation hit rate limits and reliability problems that a targeted classifier avoided entirely.
- **Genuinely non-English comments are translated, Hinglish comments are not** — this asymmetry is deliberate: the non-English bucket is small enough (~3.6%) for translation to be practical, while the Hinglish bucket (45.7%) is not.
- **Tested the app against a different kind of channel (Vsauce, a science channel)** as a control case. Toxicity was low overall, but one joke-heavy video got flagged higher — when I checked the actual comments, most were playful profanity riffing on the video's own title, not real hostility. This showed a real limitation: the model mostly looks for bad words, so it can't always tell a joke from real toxicity.
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
- Comments genuinely written in non-English scripts other than the ones tested here may still fall through to the English model if `langdetect` doesn't flag them correctly — only Hindi/Hinglish detection was specifically verified.
- The translation step for non-English comments works well when that bucket is small. For a channel where non-English content is the majority language, translating at that scale would likely hit the same rate-limit issues the Hinglish bucket did.
- The classifier conflates profanity with toxicity, since it's a lexical (word-presence) model rather than one that understands context — a transformer-based model would likely handle jokes vs. real hostility better.
- A better long-term fix for Hinglish specifically would be a multilingual model built for Indian languages (like MuRIL or XLM-RoBERTa) fine-tuned on code-mixed hate speech data, rather than a small dedicated classifier — this was out of scope here but is the clear next step.