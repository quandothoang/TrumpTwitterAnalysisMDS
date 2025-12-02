# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

"""
Data utility functions for downloading, parsing, and preprocessing Trump tweets data.

This module provides functions to:
- Download CSV data from URLs
- Parse tweets with proper handling of commas in text
- Create temporal features (time of day, season, etc.)
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


def get_season(month: int) -> str:
    """
    Convert month number to season name.
    
    Parameters
    ----------
    month : int
        Month number (1-12)
        
    Returns
    -------
    str
        Season name: 'winter', 'spring', 'summer', or 'autumn'
        
    Examples
    --------
    >>> get_season(1)
    'winter'
    >>> get_season(7)
    'summer'
    """
    if month in [1, 2, 3]:
        return "winter"
    elif month in [4, 5, 6]:
        return "spring"
    elif month in [7, 8, 9]:
        return "summer"
    else:
        return "autumn"


def get_time_of_day(hour: int) -> str:
    """
    Convert hour to time of day category.
    
    Parameters
    ----------
    hour : int
        Hour of day (0-23)
        
    Returns
    -------
    str
        Time category: 'overnight' (0-8), 'daytime' (8-16), or 'evening' (16-24)
        
    Examples
    --------
    >>> get_time_of_day(3)
    'overnight'
    >>> get_time_of_day(14)
    'daytime'
    >>> get_time_of_day(20)
    'evening'
    """
    if 0 <= hour <= 8:
        return "overnight"
    elif 8 < hour <= 16:
        return "daytime"
    else:
        return "evening"


def create_features(tweets: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal and text features from tweet data.
    
    Parameters
    ----------
    tweets : pd.DataFrame
        DataFrame with 'Date & Time' index and 'Tweet Text' column
        
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
    tweets = tweets.copy()
    
    # Text features
    tweets["length"] = tweets["Tweet Text"].str.len()
    
    # Temporal features
    tweets["hour"] = tweets.index.hour
    tweets["weekday"] = tweets.index.weekday
    tweets["year"] = tweets.index.year
    tweets["month"] = tweets.index.month
    tweets["day"] = tweets.index.day
    
    # Derived temporal features
    tweets["season"] = tweets["month"].apply(get_season)
    tweets["time_of_day"] = tweets["hour"].apply(get_time_of_day)
    
    # Additional text features
    tweets["avg_word_length"] = tweets["Tweet Text"].apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
    tweets["word_count"] = tweets["Tweet Text"].apply(lambda x: len(str(x).split()))
    tweets["punctuation_count"] = tweets["Tweet Text"].str.count(r'[^\w\s]')
    
    return tweets
