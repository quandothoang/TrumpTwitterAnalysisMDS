import pandas as pd


def load_clean_trump_csv(url):                # chatGPT assistance to construct this data-cleaning function from URL
    resp = requests.get(url)                   # download the csv text from passed URL
    resp.raise_for_status()                     # raise if 4xx/5xx
    lines = resp.text.splitlines()

    rows = []

    for i, line in enumerate(lines, start=1):
        line = line.rstrip("\n\r")
        if not line.strip():
            continue                                   # skip empty lines

        parts = line.split(",")

        if i == 1:                                        # skip header row => define custom column names later
            continue

        if len(parts) < 4:                                  # if fewer than 4 parts => it's truly broken => drop
            continue

        id_val = parts[0].strip()                              # the first 3 columns don't need cleaning
        time_val = parts[1].strip()
        url_val = parts[2].strip()
        tweet_text = ",".join(parts[3:]).strip()

        rows.append((id_val, time_val, url_val, tweet_text))

    df = pd.DataFrame(rows, columns=["ID", "Time", "Tweet URL", "Tweet Text"])
    return df


url = "https://raw.githubusercontent.com/MarkHershey/CompleteTrumpTweetsArchive/refs/heads/master/data/realDonaldTrump_in_office.csv"

tweets = load_clean_trump_csv(url)

tweets.columns = tweets.columns.str.strip()                                # strip white-space from before column names
#print(tweets.columns)
tweets["Date & Time"] = pd.to_datetime(tweets["Time"], errors="coerce")     # set Time column to DateTime and rename
tweets = tweets.drop(columns=["ID", "Tweet URL", "Time"])                     # drop ID => twitter-handle, Tweet URL, Time => now "Date & Time"

print(tweets.shape)
tweets.head(10)
