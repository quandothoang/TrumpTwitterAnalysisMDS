# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-02

# code format guidance and debugging provided by Claude Sonnet 4.5

"""
Data utility functions for Trump tweets analysis.

This module provides functions to:
- Parse raw CSV data with proper handling of commas in text
- Clean and validate tweet data
- Create categorical and numerical features (time of day, season, etc.)
- Detect outliers using IQR method
"""

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema


def parse_raw_csv(file_path):
    """
    Parses raw CSV data and handles common formatting issues.

    This function handles cases where tweet text contains commas and other
    malformed CSV rows specific to the Trump tweets dataset.

    Parameters:
    -----------
    file_path : str
        Path to the raw CSV file

    Returns:
    --------
    pd.DataFrame
        Parsed dataframe with columns: ID, Time, Tweet URL, Tweet Text
    """
    # Read the raw file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    rows = []

    # Process each line with enumeration starting at 1
    for i, line in enumerate(lines, start=1):
        # Remove trailing newline and carriage return characters
        line = line.rstrip("\n\r")

        # Skip completely empty lines (lines with only whitespace)
        if not line.strip():
            continue

        # Split the CSV line by commas into a list
        parts = line.split(",")

        # Skip the header row (first row)
        if i == 1:
            continue

        # Skip malformed rows that have fewer than 4 columns
        if len(parts) < 4:
            continue

        # Extract individual columns and strip whitespace
        id_val = parts[0].strip()
        time_val = parts[1].strip()
        url_val = parts[2].strip()

        # Handle rows where tweet text contains commas
        # Rejoin parts from index 3 onwards with commas to preserve original text
        tweet_text = ",".join(parts[3:]).strip()

        # Add this row as a tuple to the rows list
        rows.append((id_val, time_val, url_val, tweet_text))

    # Create DataFrame with proper column names
    df = pd.DataFrame(rows, columns=["ID", "Time", "Tweet URL", "Tweet Text"])
    return df


def clean_tweets(df):
    """
    Cleans and transforms the tweets dataframe.

    Performs the following operations:
    1. Strips whitespace from column names
    2. Converts Time column to datetime format
    3. Drops unnecessary columns (ID, Tweet URL, Time)
    4. Removes duplicate rows
    5. Validates data types using Pandera schema

    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe from parse_raw_csv()

    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with validated datatypes

    Raises:
    -------
    pandera.errors.SchemaError
        If dataframe fails schema validation
    """
    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()

    # Convert Time column to datetime format
    # errors='coerce' converts invalid dates to NaT (Not a Time)
    df["Date & Time"] = pd.to_datetime(df["Time"], errors="coerce")

    # Drop unnecessary columns, keeping only Tweet Text and Date & Time
    df = df.drop(columns=["ID", "Tweet URL", "Time"])

    # Remove completely duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)

    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows")
    else:
        print("No duplicate rows found")

    # Define schema for data validation
    schema = DataFrameSchema(
        {
            "Date & Time": Column(pa.DateTime, nullable=False, coerce=True),
            "Tweet Text": Column(pa.String, nullable=False),
        },
        checks=[pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")]
    )

    # Validate the dataframe against the schema
    # This will raise an error if validation fails
    df = schema.validate(df)

    print("Data validation passed")

    return df


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5):
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


def season(month: int):
    """
    This function returns the season based on the month.

    Parameters
    ----------
    month : int
        Month number (1-12)

    Returns
    -------
    str
        Season: 'winter', 'spring', 'summer', or 'autumn'

    Examples
    --------
    >>> season(1)
    'winter'
    >>> season(7)
    'summer'
    """
    if 4 <= month <= 6:
        return 'spring'
    elif 7 <= month <= 9:
        return 'summer'
    elif 10 <= month <= 12:
        return 'autumn'
    else:
        return 'winter'


def daytime(hour: int):
    """
    This function returns the time of day based on the hour.

    Parameters
    ----------
    hour : int
        Hour of day (0-23)

    Returns
    -------
    str
        Time of day: 'overnight' (0-8), 'daytime' (8-16), or 'evening' (16-24)

    Examples
    --------
    >>> daytime(3)
    'overnight'
    >>> daytime(14)
    'daytime'
    >>> daytime(20)
    'evening'
    """
    if 8 < hour <= 16:
        return 'daytime'
    elif 16 < hour <= 23 or hour == 0:
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
    for word in text.split():
        average += len(word)
    return round(average / len(text.split()), 1)


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
            count += 1
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
    tweets["word_count"] = tweets["Tweet Text"].apply(lambda x: len(x.split()))
    tweets["punctuation_count"] = tweets["Tweet Text"].apply(punctuation_count)

    return tweets


def get_raw_data_schema() -> DataFrameSchema:
    """
    Create a Pandera schema for validating raw tweet data.

    Returns
    -------
    DataFrameSchema
        Schema that validates Date & Time and Tweet Text columns
    """
    schema = DataFrameSchema(
        {
            "Date & Time": Column(pa.DateTime, nullable=False, coerce=True),
            "Tweet Text": Column(pa.String, nullable=False),
        },
        checks=[pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")]
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
