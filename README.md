# YouTube Comment Toxicity Analyzer

I built this project to check how toxic the comment section of a YouTube
channel actually is, and to see if certain videos or topics attract more
toxicity than others. I picked an Indian political commentary channel
(@SundaySarthak) for this, since I figured that kind of content would
actually have some real toxicity to find, rather than picking something
where every comment is just "nice video bro."

## What it does

1. Scrapes recent videos and comments from a YouTube channel using the
   YouTube Data API
2. Cleans and processes the comments
3. Scores each comment for toxicity (toxic, severe toxic, obscene,
   threat, insult, identity hate) using a model trained on the Jigsaw
   toxic comments dataset
4. Groups the results by video to see which videos have the highest
   toxicity rates
5. Clusters videos based on toxicity, views, and likes to look for
   patterns in what drives toxic comment sections
6. A Streamlit app where you can type in any channel handle and get a
   live toxicity report

## A problem I ran into (and how I dealt with it)

This is the part of the project I'm actually most proud of, even though
it's not something that "worked perfectly."

The toxicity model is trained on Jigsaw's dataset, which is all English.
But when I actually looked at the comments I scraped, I found that
almost half of them (45.7%, to be exact) were in Hinglish — Hindi
written in English letters, like "yeh video bahut acha hai." A lot of
Indian YouTube comments are like this.

The problem is that a tool called `langdetect`, which is supposed to
figure out what language a piece of text is in, could not handle this
at all. It kept guessing completely random languages for these
comments — Indonesian, Somali, Swahili, Estonian, even Welsh. Obviously
none of that is right. What's actually happening is that short romanized
Hindi words just happen to share some letter patterns with those
languages, and the tool has no way to tell.

So I tried a few things to fix this:
- Translating everything to English first (Google Translate) — this hit
  rate limits and took over an hour to run on my data, which wasn't
  practical
- Converting to Hindi script first, then translating — the conversion
  itself wasn't reliable, it sometimes turned English words like "video"
  into gibberish
- A couple of free/offline translation tools — didn't get these working
  in time

None of these were going to work well enough, so instead I trained a
**second, separate model** just for Hinglish. I found a small dataset
(about 1,366 comments) of Hinglish text labeled as offensive or not, and
trained a basic classifier on it. Then I built simple routing logic: if
a comment is flagged as likely Hinglish, it goes through this model
instead of the English one.

This actually made a big difference. Without the Hinglish model, my
pipeline found 331 toxic comments out of 12,521. With it, that jumped
to 774 — more than double. That means the English-only version was
missing a lot of real toxic comments just because it couldn't
understand the language they were written in.

I think this is a more honest and realistic result than if I'd just
ignored the problem, or pretended a quick translation fix solved
everything.

## What I found

- Toxicity varies a lot by video, even on the same channel — from about
  1.8% to nearly 19% of comments flagged toxic
- The channel's category is almost entirely "News & Politics," so that
  alone doesn't explain the differences
- When I clustered the videos, the pattern was pretty clear: videos with
  angry, accusatory titles about one specific incident ("Shameful",
  "Why Are Reporters Targeted?") had way more toxic comments than the
  multi-topic weekly roundup videos, even when the roundup videos got
  similar or higher views
- I also tested the app on a completely different kind of channel
  (Vsauce, a science channel) to see how it handled a calmer comment
  section. Toxicity was low overall (2-6%) on most videos, but one
  video with a swearing-related title got flagged a lot higher (16%).
  When I actually read the flagged comments, most of them weren't
  actually toxic — they were jokes playing on the video's own title.
  This showed me a real weakness of this kind of model: it mostly
  looks for bad words, so it can't always tell the difference between
  someone joking around and someone actually being hostile.

## Tech I used

Python, pandas, scikit-learn (TF-IDF, Logistic Regression, Linear SVC,
KMeans), YouTube Data API v3, langdetect, Streamlit, Plotly

## What I'd do differently with more time

- Use a better model for Hinglish specifically — something like MuRIL
  or XLM-RoBERTa, which are actually built for Indian languages, instead
  of a small Logistic Regression model
- Add more labeled Hinglish data — 1,366 examples got me a working
  model, but a bigger dataset would help it generalize better
- Try to fix the "profanity vs. actual toxicity" problem, maybe by using
  a model that understands context better instead of one that mostly
  looks at word presence
- Test on more channels to see if the "angry single-topic videos are
  more toxic than roundups" pattern holds up elsewhere, or if it's just
  specific to this channel

## How to run it

```bash
pip install -r requirements.txt
```

You'll need a YouTube Data API key (free from Google Cloud Console).
Put it in a `.env` file:
```
YOUTUBE_API_KEY=your_key_here
```

Run the notebooks in order (01, 02, 03) to reproduce the models, or run
the app directly if the models are already trained:
```bash
streamlit run app.py
```****