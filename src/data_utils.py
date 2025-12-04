# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

"""
Data utility functions for downloading, parsing, and preprocessing Trump tweets data.

This module provides functions to:
- Download CSV data from URLs
- Parse tweets with proper handling of commas in text
- Create categorical and numerical features (time of day, season, etc.)
- Validate data with Pandera schemas
- Detect outliers using IQR method
"""

import pandas as pd
import numpy as np
import requests
import pandera as pa
from pandera import Column, Check, DataFrameSchema


def download_and_parse_csv(url: str) -> pd.DataFrame:
    """
    Download Trump tweets CSV from URL and parse it.
    
    The original CSV has commas within tweet text, so we handle this
    by joining all fields after the first 3 columns as the tweet text.
    
    Parameters
    ----------
    url : str
        URL to the raw CSV file
        
    Returns
    -------
    pd.DataFrame
        Dataframe with columns: Tweet Text, Date & Time
        
    Examples
    --------
    >>> url = "https://raw.githubusercontent.com/.../realDonaldTrump_in_office.csv"
    >>> df = download_and_parse_csv(url)
    >>> df.columns.tolist()
    ['Tweet Text', 'Date & Time']
    """
    print(f"Fetching data from URL: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    print(f"Successfully downloaded {len(resp.content)} bytes")
    
    lines = resp.text.splitlines()
    rows = []

    for i, line in enumerate(lines, start=1):
        line = line.rstrip("\n\r")
        if not line.strip():
            continue

        parts = line.split(",")

        # Skip header row
        if i == 1:
            continue

        if len(parts) < 4:
            continue

        id_val = parts[0].strip()
        time_val = parts[1].strip()
        url_val = parts[2].strip()
        # Handle commas in tweet text by joining remaining parts
        tweet_text = ",".join(parts[3:]).strip()

        rows.append((id_val, time_val, url_val, tweet_text))

    df = pd.DataFrame(rows, columns=["ID", "Time", "Tweet URL", "Tweet Text"])
    
    # Clean and convert to datetime
    df.columns = df.columns.str.strip()
    df["Date & Time"] = pd.to_datetime(df["Time"], errors="coerce")
    df = df.drop(columns=["ID", "Tweet URL", "Time"])
    
    return df


def get_raw_data_schema() -> DataFrameSchema:
    """
    Create a Pandera schema for validating raw tweet data.
    
    Returns
    -------
    pa.DataFrameSchema
        Schema that validates:
        - Date & Time: datetime64[ns], no nulls
        - Tweet Text: string type, no nulls
        - No completely empty rows
        
    Examples
    --------
    >>> schema = get_raw_data_schema()
    >>> validated_df = schema.validate(raw_df)
    """
    schema = DataFrameSchema(
        {
            "Date & Time": Column(
                pa.DateTime,
                nullable=False,
                coerce=True,
                checks=[
                    Check(lambda s: s.notna().all(), error="Date & Time contains null values")
                ]
            ),
            "Tweet Text": Column(
                str,
                nullable=False,
                checks=[
                    Check(lambda s: s.str.len() > 0, error="Tweet Text contains empty strings")
                ]
            ),
        },
        checks=[
            Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
        ],
        strict=False,  # Allow additional columns
        coerce=True
    )
    return schema


def validate_data(tweets: pd.DataFrame) -> pd.DataFrame:
    """
    Validate tweet data using Pandera schema.
    
    Parameters
    ----------
    tweets : pd.DataFrame
        DataFrame with 'Date & Time' and 'Tweet Text' columns
        
    Returns
    -------
    pd.DataFrame
        Validated DataFrame
        
    Raises
    ------
    pandera.errors.SchemaError
        If validation fails
        
    Examples
    --------
    >>> validated_df = validate_data(tweets)
    >>> print("Validation passed!")
    """
    schema = get_raw_data_schema()
    return schema.validate(tweets)


def check_datetime_info(tweets: pd.DataFrame) -> None:
    """
    Check whether datetime exists as column or index and print info.
    
    Parameters
    ----------
    tweets : pd.DataFrame
        DataFrame to check
    """
    if isinstance(tweets.index, pd.DatetimeIndex):
        print("Datetime index detected — dates are correctly parsed as index.")
    elif "Date & Time" in tweets.columns:
        print("'Date & Time' column exists — first few values:")
        print(tweets["Date & Time"].head())
    else:
        print("No datetime information found (neither column nor index).")


def remove_duplicates(tweets: pd.DataFrame, subset: list = None) -> tuple:
    """
    Remove duplicate rows from DataFrame.
    
    Parameters
    ----------
    tweets : pd.DataFrame
        DataFrame to deduplicate
    subset : list, optional
        Columns to consider for duplicates. If None, uses all columns.
        
    Returns
    -------
    tuple
        (deduplicated DataFrame, number of duplicates removed)
        
    Examples
    --------
    >>> df_clean, n_removed = remove_duplicates(tweets, subset=["Tweet Text"])
    >>> print(f"Removed {n_removed} duplicates")
    """
    original_count = len(tweets)
    if subset:
        tweets_clean = tweets.drop_duplicates(subset=subset)
    else:
        tweets_clean = tweets.drop_duplicates()
    duplicates_removed = original_count - len(tweets_clean)
    return tweets_clean, duplicates_removed


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> tuple:
    """
    Detect outliers using the IQR (Interquartile Range) method.
    
    Parameters
    ----------
    series : pd.Series
        Numeric series to check for outliers
    multiplier : float, default=1.5
        IQR multiplier for bounds (1.5 is standard)
        
    Returns
    -------
    tuple
        (outlier_mask, lower_bound, upper_bound, outlier_count)
        
    Examples
    --------
    >>> mask, lower, upper, count = detect_outliers_iqr(tweets["length"])
    >>> print(f"Found {count} outliers outside [{lower}, {upper}]")
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_count = outlier_mask.sum()
    
    return outlier_mask, lower_bound, upper_bound, outlier_count


def season(date: pd.Timestamp):
    """
    This function returns the season based on the month.
    
    Parameters
    ----------
    date : pd.Timestamp
        The date we want to find the season from.
        
    Returns
    -------
    str
        Season: 'winter', 'spring', 'summer', or 'autumn'
        
    Examples
    --------
    >>> get_season(1)
    'winter'
    >>> get_season(7)
    'summer'
    """
    if 4 <= date.month <=6:
        return 'spring'
    elif 7 <= date.month <=9:
        return 'spring'
    elif 10 <= date.month <=12:
        return 'autumn'
    else:
        return 'winter'


def daytime(date: pd.Timestamp):
    """
    This function returns the time of day based on the hour.
    
    Parameters
    ----------
    date : pd.Timestamp
        The date we want to find the time of day from.
        
    Returns
    -------
    str
        Time of day: 'overnight' (0-8), 'daytime' (8-16), or 'evening' (16-24)
        
    Examples
    --------
    >>> get_time_of_day(3)
    'overnight'
    >>> get_time_of_day(14)
    'daytime'
    >>> get_time_of_day(20)
    'evening'
    """
    if pd.Timestamp('08:01').time() <= date.time() <= pd.Timestamp('16:00').time():
        return 'daytime'
    elif pd.Timestamp('16:01').time() <= date.time() <= pd.Timestamp('00:00').time():
        return 'evening'
    else:
        return 'overnight'
    
def avg_word_length(text: str):
    """
    This function finds the average word length in a text. 
    
    Parameters
    ----------
    text : str
        A text where each word is separated by spaces.
        
    Returns
    -------
    float
        The average word length rounded to 1 decimal point.
        
    Examples
    --------
    >>> avg_word_length('Donald Trump first presidency began in January 2017, and ended in January 2021.')
    5.2
    """
    average = 0
    for word in text.split() :
        average += len(word)
    return round(average/len(text.split()),1)

def punctuation_count(text):
    """
    This function finds the number of punctuation marks in the text. Here, punctuation marks are considered to be any non-numeric or whitespace character (this includes symbols like & or #).
    
    Parameters
    ----------
    text : str
        A text.
        
    Returns
    -------
    int
        The number of punctuation marks.
        
    Examples
    --------
    >>> punctuation_count('Donald Trump first presidency began in January 2017, and ended in January 2021.')
    2
    """
    count = 0
    for char in text:
        if not char.isalnum() and not char.isspace():
            count+=1
    return count


def create_features(tweets: pd.DataFrame):
    """
    This function creates categorical and numerical features from the existing features.
    
    Parameters
    ----------
    tweets : pd.DataFrame
        DataFrame with Date & Time and Tweet Text columns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional feature columns:
        - length: character count of tweet
        - hour, weekday, year, month, day: temporal components
        - season: winter/spring/summer/autumn
        - time_of_day: overnight/daytime/evening
        - avg_word_length: average word length in tweet
        - word_count: number of words
        - punctuation_count: count of punctuation marks
        
    Examples
    --------
    >>> df = create_features(tweets)
    >>> 'season' in df.columns
    True
    """
    tweets = tweets.reset_index().copy()
    
    # Text features
    tweets["length"] = tweets["Tweet Text"].str.len()
    
    # Temporal features
    tweets["hour"] = tweets['Date & Time'].dt.hour
    tweets["weekday"] = tweets['Date & Time'].dt.weekday
    tweets["year"] = tweets['Date & Time'].dt.year
    tweets["month"] = tweets['Date & Time'].dt.month
    tweets["day"] = tweets['Date & Time'].dt.day
    
    # Derived temporal features
    tweets["season"] = tweets["month"].apply(season)
    tweets["time_of_day"] = tweets["hour"].apply(daytime)
    
    # Additional text features
    tweets["avg_word_length"] = tweets["Tweet Text"].apply(avg_word_length) 
    tweets["word_count"] = tweets["Tweet Text"].apply(lambda x:len(x.split()))
    tweets["punctuation_count"] = tweets["Tweet Text"].apply(punctuation_count)
    
    return tweets
