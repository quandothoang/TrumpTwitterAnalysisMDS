def main():
    """
    Handle null and duplicate values and validate datatypes.
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandera.pandas as pa
    from pandera.pandas import Column, DataFrameSchema
    from load_data_and_clean import load_clean_trump_csv
    
    url = "https://raw.githubusercontent.com/MarkHershey/CompleteTrumpTweetsArchive/master/data/realDonaldTrump_in_office.csv"
    tweets = load_clean_trump_csv(url)
    tweets.columns = tweets.columns.str.strip()
    tweets["Date & Time"] = pd.to_datetime(tweets["Time"], errors="coerce")
    tweets = tweets.drop(columns=["ID", "Tweet URL", "Time"])
    
    # Define schema for validation
    schema = DataFrameSchema(
        {
            "Date & Time": Column(pa.DateTime, nullable=False, coerce=True),
            "Tweet Text": Column(pa.String, nullable=False),
        },
        checks=[pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")]
    )
    
    tweets_valid = schema.validate(tweets)
    print("Column data types:")
    print(tweets.dtypes)
    
    # Check whether datetime exists either as a column OR as index
    if isinstance(tweets.index, pd.DatetimeIndex):
        print("\nDatetime index detected — dates are correctly parsed as index.")
    else:
        if "Date & Time" in tweets.columns:
            print("\n'Date & Time' column exists — check first few values:")
            print(tweets["Date & Time"].head())
        else:
            print("\nNo datetime information found (neither column nor index).")
    
    # Check for duplicates
    dup_count = tweets.duplicated().sum()
    print(f"Number of duplicated rows: {dup_count}")
    
    if dup_count > 0:
        tweets = tweets.drop_duplicates()
        print("Duplicates removed.")
    else:
        print("No duplicate observations found.")
    
    # Create length feature
    tweets["length"] = tweets["Tweet Text"].str.len()
    print("\nLength descriptive statistics:")
    print(tweets["length"].describe())
    
    # Boxplot to visualize outliers
    sns.boxplot(data=tweets, x="length")
    plt.title("Outlier Check — Tweet Length")
    plt.show()
    #plt.savefig('tweet_length_outliers.png')              # uncomment to save plot as .png 
    #plt.close() 
    # print("Plot saved as 'tweet_length_outliers.png'")
    
    # Using IQR rule to detect outliers
    Q1 = tweets["length"].quantile(0.25)
    Q3 = tweets["length"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = tweets[(tweets["length"] < lower_bound) | (tweets["length"] > upper_bound)]
    print(f"\nNumber of detected outliers: {outliers.shape[0]}")

if __name__ == "__main__":  
    main()
    print("Null values and duplicates handled and row datatypes passed validation!")