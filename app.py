import html
import os
import re
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from googleapiclient.discovery import build
from langdetect import detect, DetectorFactory

load_dotenv()
DetectorFactory.seed = 0

st.set_page_config(page_title="YouTube Channel Toxicity Scanner", page_icon="🔍", layout="wide")

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
PLAUSIBLE_NON_ENGLISH = {"hi", "mr", "bn", "pa", "ne", "zh-cn", "vi", "ta", "te", "gu"}

# ---------------------------------------------------------------
# Load all four trained artifacts: Jigsaw model + Hinglish model
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
    classifier = joblib.load("models/toxicity_classifier.joblib")
    thresholds = joblib.load("models/optimal_thresholds.joblib")
    hinglish_vectorizer = joblib.load("models/hinglish_vectorizer.joblib")
    hinglish_classifier = joblib.load("models/hinglish_classifier.joblib")
    return vectorizer, classifier, thresholds, hinglish_vectorizer, hinglish_classifier

vectorizer, classifier, thresholds, hinglish_vectorizer, hinglish_classifier = load_artifacts()

# ---------------------------------------------------------------
# YouTube fetch helpers
# ---------------------------------------------------------------
def resolve_channel(youtube, handle):
    handle = handle.strip()
    if not handle.startswith("@"):
        handle = "@" + handle
    res = youtube.channels().list(part="snippet,contentDetails", forHandle=handle).execute()
    if not res.get("items"):
        raise ValueError(f"Channel '{handle}' not found.")
    item = res["items"][0]
    return item["snippet"]["title"], item["contentDetails"]["relatedPlaylists"]["uploads"]

def fetch_recent_videos(youtube, uploads_id, n):
    res = youtube.playlistItems().list(
        part="snippet,contentDetails", playlistId=uploads_id, maxResults=min(n, 50)
    ).execute()
    video_ids = [it["contentDetails"]["videoId"] for it in res.get("items", [])]

    # Check live/upcoming status — live streams and premieres often have
    # comments disabled or unavailable via commentThreads until they end
    status_res = youtube.videos().list(part="snippet", id=",".join(video_ids)).execute()
    live_status = {item["id"]: item["snippet"].get("liveBroadcastContent", "none") for item in status_res.get("items", [])}

    return [
        {
            "video_id": it["contentDetails"]["videoId"],
            "title": it["snippet"]["title"],
            "live_status": live_status.get(it["contentDetails"]["videoId"], "none"),
        }
        for it in res.get("items", [])
    ]

def fetch_comments(youtube, video_id, n_per_mode):
    comments = []
    for order in ["relevance", "time"]:
        try:
            res = youtube.commentThreads().list(
                part="snippet", videoId=video_id, order=order,
                textFormat="plainText", maxResults=min(n_per_mode, 100)
            ).execute()
            for item in res.get("items", []):
                c = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({"video_id": video_id, "raw_text": c["textOriginal"]})
        except Exception:
            continue  # comments disabled on this video
    return comments

# ---------------------------------------------------------------
# Cleaning + language routing (same logic validated in the notebooks)
# ---------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def language_status(text):
    if len(text) < 3:
        return "too_short"
    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"
    if lang == "en":
        return "english"
    if lang in PLAUSIBLE_NON_ENGLISH:
        return "likely_genuine_non_english"
    return "likely_misdetected_hinglish"

def score_comment(text, status):
    if status == "likely_misdetected_hinglish":
        vec = hinglish_vectorizer.transform([text])
        proba = hinglish_classifier.predict_proba(vec)[0][1]
        return "hinglish", int(proba >= 0.5)
    else:
        vec = vectorizer.transform([text])
        probs = classifier.predict_proba(vec)[0]
        is_toxic = int(any(probs[i] >= thresholds[col] for i, col in enumerate(LABEL_COLS)))
        return "jigsaw", is_toxic

# ---------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------
st.sidebar.title("⚙️ Scanner Settings")
api_key = st.sidebar.text_input("YouTube API Key", value=os.getenv("YOUTUBE_API_KEY", ""), type="password")
channel_input = st.sidebar.text_input("YouTube Channel Handle", placeholder="@channelname")
n_videos = st.sidebar.slider("Videos to scan", 2, 15, 5)
n_comments = st.sidebar.slider("Comments per video", 20, 100, 50, step=10)
scan_button = st.sidebar.button("🚀 Scan Channel", type="primary", use_container_width=True)

st.title("🔍 YouTube Channel Toxicity Scanner")
st.caption("Scores comments using an English toxicity model (Jigsaw) and a dedicated Hinglish "
           "offensiveness model, routed by detected language — see README for why this matters.")

if scan_button:
    if not api_key:
        st.error("Enter a YouTube API key in the sidebar.")
        st.stop()
    if not channel_input:
        st.error("Enter a channel handle.")
        st.stop()

    try:
        youtube = build("youtube", "v3", developerKey=api_key)

        with st.status(f"Scanning {channel_input}...", expanded=True) as status:
            st.write("Resolving channel...")
            channel_title, uploads_id = resolve_channel(youtube, channel_input)

            st.write(f"Fetching {n_videos} recent videos...")
            videos = fetch_recent_videos(youtube, uploads_id, n_videos)

            st.write("Fetching comments...")
            rows = []
            skipped_live = 0
            for v in videos:
                if v["live_status"] != "none":
                    skipped_live += 1
                    continue
                for c in fetch_comments(youtube, v["video_id"], n_comments // 2):
                    c["video_title"] = v["title"]
                    rows.append(c)

            if skipped_live:
                st.caption(f"Skipped {skipped_live} live/upcoming video(s) — comments unavailable during broadcast.")

            if not rows:
                st.warning("No comments retrieved (comments may be disabled).")
                st.stop()

            df = pd.DataFrame(rows)
            df["cleaned_text"] = df["raw_text"].apply(clean_text)
            df["language_status"] = df["cleaned_text"].apply(language_status)

            st.write("Scoring comments (routed by language)...")
            results = df.apply(
                lambda r: score_comment(r["cleaned_text"], r["language_status"]), axis=1
            )
            df["model_used"] = [r[0] for r in results]
            df["is_toxic"] = [r[1] for r in results]

            status.update(label=f"Scan complete: {channel_title}", state="complete", expanded=False)

        # -----------------------------------------------------------
        # Results
        # -----------------------------------------------------------
        st.header(f"📊 Results: {channel_title}")

        total = len(df)
        toxic = df["is_toxic"].sum()
        hinglish_pct = (df["language_status"] == "likely_misdetected_hinglish").mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Videos Scanned", len(videos))
        col2.metric("Comments Scored", f"{total:,}")
        col3.metric("Toxicity Rate", f"{toxic / total * 100:.1f}%")
        col4.metric("Likely Hinglish", f"{hinglish_pct:.0f}%")

        st.divider()

        video_summary = (
            df.groupby("video_title")
            .agg(comments=("is_toxic", "count"), toxic=("is_toxic", "sum"))
            .reset_index()
        )
        video_summary["toxicity_rate"] = (video_summary["toxic"] / video_summary["comments"] * 100).round(1)
        video_summary = video_summary.sort_values("toxicity_rate")

        fig = px.bar(
            video_summary, x="toxicity_rate", y="video_title", orientation="h",
            color="toxicity_rate", color_continuous_scale="Reds", text_auto=".1f",
            labels={"toxicity_rate": "Toxicity Rate (%)", "video_title": "Video"},
        )
        fig.update_layout(yaxis_title=None, xaxis_ticksuffix="%", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Flagged Comments")
        flagged = df[df["is_toxic"] == 1][["video_title", "cleaned_text", "model_used", "language_status"]].head(50)
        if not flagged.empty:
            st.dataframe(flagged, use_container_width=True)
        else:
            st.success("No toxic comments detected in this batch.")

    except Exception as e:
        st.error(f"Scan failed: {e}")
else:
    st.info("Enter a channel handle in the sidebar and click **Scan Channel** to run a live audit.")