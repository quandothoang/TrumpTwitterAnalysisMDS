# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

"""
Visualization utility functions for Trump tweets analysis.

This module provides functions to:
- Create Altair charts for EDA
- Create sentiment distribution and time series charts
- Generate word clouds
"""

import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def create_time_of_day_chart(tweets: pd.DataFrame) -> alt.LayerChart:
    """
    Create a bar chart showing tweet frequency by time of day.
    
    Parameters
    ----------
    tweets : pd.DataFrame
        DataFrame with 'time_of_day' column
        
    Returns
    -------
    alt.LayerChart
        Altair chart with bars and text annotations
        
    Examples
    --------
    >>> chart = create_time_of_day_chart(tweets)
    >>> chart.save("time_of_day.png")
    """
    time_counts = tweets["time_of_day"].value_counts().reset_index()
    time_counts.columns = ["Time", "Count"]
    
    time_ranges = {
        "daytime": "8:01am – 4:00pm",
        "evening": "4:01pm – 12:00am", 
        "overnight": "12:01am – 8:00am"
    }
    time_counts["Time Range"] = time_counts["Time"].map(time_ranges)
    
    total = time_counts["Count"].sum()
    time_counts["Percentage"] = (time_counts["Count"] / total * 100).round(1)
    
    time_bars = alt.Chart(time_counts).mark_bar(color="#c0392b").encode(
        y=alt.Y("Time:N", 
                title="Time of Day",
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        x=alt.X("Count:Q", 
                title="Number of Tweets",
                axis=alt.Axis(labelFontSize=12, titleFontSize=16)),
        tooltip=[
            alt.Tooltip("Time:N", title="Time Period"),
            alt.Tooltip("Time Range:N", title="Hours"),
            alt.Tooltip("Count:Q", title="Tweet Count", format=","),
            alt.Tooltip("Percentage:Q", title="Percentage", format=".1f")
        ]
    ).properties(
        title=alt.TitleParams(
            text="Trump's Tweet Frequency by Time of Day",
            fontSize=18,
            anchor="middle"
        ),
        width=550,
        height=300
    )
    
    range_text = alt.Chart(time_counts).mark_text(
        align="left",
        baseline="middle",
        color="white",
        fontSize=14,
        fontWeight="bold",
        dx=10
    ).encode(
        y="Time:N",
        x=alt.value(5),
        text="Time Range:N"
    )
    
    count_text = alt.Chart(time_counts).mark_text(
        align="left",
        baseline="middle",
        dx=5,
        color="black",
        fontSize=13,
        fontWeight="bold"
    ).encode(
        y="Time:N",
        x="Count:Q",
        text=alt.Text("Count:Q", format=",")
    )
    
    return (time_bars + range_text + count_text).properties(
        padding={"right": 70, "top": 20, "bottom": 20, "left": 10}
    ).configure_axis(
        labelFontSize=13,
        titleFontSize=15
    )


def create_seasonal_chart(tweets: pd.DataFrame) -> alt.LayerChart:
    """
    Create a bar chart showing tweet frequency by season.
    
    Parameters
    ----------
    tweets : pd.DataFrame
        DataFrame with 'season' column
        
    Returns
    -------
    alt.LayerChart
        Altair chart with bars and text annotations
        
    Examples
    --------
    >>> chart = create_seasonal_chart(tweets)
    >>> chart.save("seasonal.png")
    """
    season_counts = tweets["season"].value_counts().reset_index()
    season_counts.columns = ["Season", "Count"]
    season_counts["Season"] = season_counts["Season"].str.capitalize()
    
    season_ranges = {
        "Spring": "Apr 1 – Jun 30",
        "Summer": "Jul 1 – Sep 30",
        "Autumn": "Oct 1 – Dec 31",
        "Winter": "Jan 1 – Mar 31"
    }
    season_counts["Date Range"] = season_counts["Season"].map(season_ranges)
    
    total = season_counts["Count"].sum()
    season_counts["Percentage"] = (season_counts["Count"] / total * 100).round(1)
    
    season_bars = alt.Chart(season_counts).mark_bar(color="#c0392b").encode(
        x=alt.X("Count:Q",
                title="Number of Tweets",
                axis=alt.Axis(labelFontSize=12, titleFontSize=16)),
        y=alt.Y("Season:N", 
                title="Season",
                sort=["Spring", "Summer", "Autumn", "Winter"],
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        tooltip=[
            alt.Tooltip("Season:N", title="Season"),
            alt.Tooltip("Date Range:N", title="Months"),
            alt.Tooltip("Count:Q", title="Tweet Count", format=","),
            alt.Tooltip("Percentage:Q", title="Percentage", format=".1f")
        ]
    ).properties(
        title=alt.TitleParams(
            text="Trump's Tweet Frequency by Season",
            fontSize=18,
            anchor="middle"
        ),
        width=550,
        height=300
    )
    
    range_text = alt.Chart(season_counts).mark_text(
        align="left",
        baseline="middle",
        color="white",
        fontSize=14,
        fontWeight="bold",
        dx=10
    ).encode(
        y=alt.Y("Season:N", sort=["Spring", "Summer", "Autumn", "Winter"]),
        x=alt.value(5),
        text="Date Range:N"
    )
    
    count_text = alt.Chart(season_counts).mark_text(
        align="left",
        baseline="middle",
        dx=5,
        color="black",
        fontSize=13,
        fontWeight="bold"
    ).encode(
        y=alt.Y("Season:N", sort=["Spring", "Summer", "Autumn", "Winter"]),
        x="Count:Q",
        text=alt.Text("Count:Q", format=",")
    )
    
    return (season_bars + range_text + count_text).properties(
        padding={"right": 70, "top": 20, "bottom": 20, "left": 10}
    ).configure_axis(
        labelFontSize=13,
        titleFontSize=15
    )


def create_sentiment_chart(tweets: pd.DataFrame) -> alt.Chart:
    """
    Create a bar chart showing sentiment distribution.
    
    Parameters
    ----------
    tweets : pd.DataFrame
        DataFrame with 'Sentiment' column
        
    Returns
    -------
    alt.Chart
        Altair chart with colored bars by sentiment
        
    Examples
    --------
    >>> chart = create_sentiment_chart(tweets)
    >>> chart.save("sentiment.png")
    """
    sentiment_counts = tweets["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    total = sentiment_counts["Count"].sum()
    sentiment_counts["Percentage"] = (sentiment_counts["Count"] / total * 100).round(1)
    
    order = ["positive", "neutral", "negative"]
    sentiment_counts["order"] = sentiment_counts["Sentiment"].map(
        {s: i for i, s in enumerate(order)}
    )
    sentiment_counts = sentiment_counts.sort_values("order")
    
    bars = alt.Chart(sentiment_counts).mark_bar().encode(
        x=alt.X("Sentiment:N", 
                title="Sentiment Category",
                sort=order,
                axis=alt.Axis(labelFontSize=14, titleFontSize=16, labelAngle=0)),
        y=alt.Y("Count:Q", 
                title="Number of Tweets",
                axis=alt.Axis(labelFontSize=12, titleFontSize=16)),
        color=alt.Color("Sentiment:N", 
                       scale=alt.Scale(
                           domain=["positive", "neutral", "negative"],
                           range=["#27ae60", "#95a5a6", "#e74c3c"]
                       ),
                       legend=alt.Legend(
                           title="Sentiment",
                           titleFontSize=14,
                           labelFontSize=12,
                           orient="right"
                       )),
        tooltip=[
            alt.Tooltip("Sentiment:N", title="Sentiment"),
            alt.Tooltip("Count:Q", title="Count", format=","),
            alt.Tooltip("Percentage:Q", title="Percentage", format=".1f")
        ]
    ).properties(
        title=alt.TitleParams(
            text="Distribution of Tweet Sentiments",
            subtitle="Classified using VADER sentiment analysis (threshold: ±0.05)",
            fontSize=18,
            subtitleFontSize=12
        ),
        width=400,
        height=350
    )
    
    text = alt.Chart(sentiment_counts).mark_text(
        dy=-10,
        fontSize=13,
        fontWeight="bold",
        color="black"
    ).encode(
        x=alt.X("Sentiment:N", sort=order),
        y="Count:Q",
        text=alt.Text("Count:Q", format=",")
    )
    
    return (bars + text).configure_axis(
        labelFontSize=13,
        titleFontSize=15
    )


def create_sentiment_over_time_chart(tweets: pd.DataFrame) -> alt.Chart:
    """
    Create a line chart showing sentiment trends over time.
    
    Parameters
    ----------
    tweets : pd.DataFrame
        DataFrame with 'Sentiment' column and datetime index
        
    Returns
    -------
    alt.Chart
        Altair line chart with colored lines by sentiment
        
    Examples
    --------
    >>> chart = create_sentiment_over_time_chart(tweets)
    >>> chart.save("sentiment_time.png")
    """
    monthly_sentiment = tweets.groupby([pd.Grouper(freq='M'), 'Sentiment']).size().unstack(fill_value=0)
    monthly_sentiment = monthly_sentiment.reset_index()
    monthly_sentiment = monthly_sentiment.melt(
        id_vars=['Date & Time'], 
        var_name='Sentiment', 
        value_name='Count'
    )
    
    chart = alt.Chart(monthly_sentiment).mark_line(
        point=alt.OverlayMarkDef(size=50),
        strokeWidth=2
    ).encode(
        x=alt.X('Date & Time:T', 
                title='Date',
                axis=alt.Axis(labelFontSize=11, titleFontSize=14, format='%b %Y')),
        y=alt.Y('Count:Q', 
                title='Number of Tweets per Month',
                axis=alt.Axis(labelFontSize=11, titleFontSize=14)),
        color=alt.Color('Sentiment:N', 
                       scale=alt.Scale(
                           domain=['positive', 'neutral', 'negative'],
                           range=['#27ae60', '#95a5a6', '#e74c3c']
                       ),
                       legend=alt.Legend(
                           title="Sentiment",
                           titleFontSize=14,
                           labelFontSize=12,
                           orient="right"
                       )),
        tooltip=[
            alt.Tooltip('Date & Time:T', title='Month', format='%B %Y'),
            alt.Tooltip('Sentiment:N', title='Sentiment'),
            alt.Tooltip('Count:Q', title='Tweet Count', format=',')
        ]
    ).properties(
        title=alt.TitleParams(
            text='Monthly Sentiment Trends (2017-2021)',
            subtitle='Number of positive, neutral, and negative tweets per month',
            fontSize=18,
            subtitleFontSize=12
        ),
        width=750,
        height=400
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    
    return chart


def create_wordcloud(word_freqs: dict, title: str, colormap: str = "Greens") -> plt.Figure:
    """
    Create a word cloud from word frequencies.
    
    Parameters
    ----------
    word_freqs : dict
        Dictionary of word -> frequency/weight
    title : str
        Title for the word cloud
    colormap : str, default="Greens"
        Matplotlib colormap name
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with word cloud
        
    Examples
    --------
    >>> fig = create_wordcloud({"great": 0.5, "good": 0.3}, "Positive Words")
    >>> fig.savefig("wordcloud.png")
    """
    if not word_freqs:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "No words to display", ha='center', va='center', fontsize=16)
        ax.axis("off")
        return fig
    
    wc = WordCloud(
        width=1000,
        height=500,
        background_color="white",
        colormap=colormap,
        max_words=75,
        min_font_size=12,
        max_font_size=120,
        prefer_horizontal=0.9,
        relative_scaling=0.5
    ).generate_from_frequencies(word_freqs)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
    plt.tight_layout()
    
    return fig
