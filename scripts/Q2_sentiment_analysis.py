
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def main():
    # Load data
    tweets = pd.read_csv("trump_tweets.csv")  # <-- change filename if needed

    # Sentiment score
    sia = SentimentIntensityAnalyzer()
    tweets["Tweet Text"] = tweets["Tweet Text"].fillna("").astype(str)
    tweets["sentiment_score"] = tweets["Tweet Text"].apply(
        lambda t: sia.polarity_scores(t)["compound"]
    )

    # Convert to labels
    def score_to_label(score, pos_threshold=0.05, neg_threshold=-0.05):
        if score >= pos_threshold:
            return "positive"
        elif score <= neg_threshold:
            return "negative"
        else:
            return "neutral"

    tweets["sentiment_label"] = tweets["sentiment_score"].apply(score_to_label)

    # Counts
    sentiment_counts = (
        tweets.groupby("sentiment_label")
        .size()
        .reset_index(name="Count")
        .rename(columns={"sentiment_label": "Sentiment"})
    )
    print(sentiment_counts)

    # Chart
    chart = (
        alt.Chart(sentiment_counts)
        .mark_bar()
        .encode(
            x=alt.X("Sentiment:N", sort=["positive", "neutral", "negative"]),
            y=alt.Y("Count:Q"),
            tooltip=["Sentiment", "Count"]
        )
        .properties(title="Number of Positive, Neutral, and Negative Tweets")
    )

    chart.save("sentiment_chart.html")
    print("Saved: sentiment_chart.html")


if __name__ == "__main__":
    main()

