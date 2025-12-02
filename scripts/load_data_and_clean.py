def main():

    import pandas as pd
    import altair as alt
    import requests
    import pandera.pandas as pa
    from pandera.pandas import Column, DataFrameSchema
    def load_clean_trump_csv(url):
        resp = requests.get(url)
        resp.raise_for_status()
        lines = resp.text.splitlines()
        rows = []
        for i, line in enumerate(lines, start=1):
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
                parts = line.split(",")
            if i == 1:
                if len(parts) < 4:
                    continue
                id_val = parts[0].strip()
                time_val = parts[1].strip()
                url_val = parts[2].strip()
        tweet_text = ",".join(parts[3:]).strip()
        rows.append((id_val, time_val, url_val, tweet_text))
        df = pd.DataFrame(rows, columns=["ID", "Time", "Tweet URL", "Tweet Text"])
        return df
    url = "https://raw.githubusercontent.com/MarkHershey/CompleteTrumpTweetsArch"
    tweets = load_clean_trump_csv(url)
    tweets.columns = tweets.columns.str.strip()
    tweets["Date & Time"] = pd.to_datetime(tweets["Time"], errors="coerce")
    tweets = tweets.drop(columns=["ID", "Tweet URL", "Time"])
    print(tweets.shape)
    tweets.head(10)
    print("Successfully Loaded and Cleaned Data")


if __name__ == "main":
    main()

