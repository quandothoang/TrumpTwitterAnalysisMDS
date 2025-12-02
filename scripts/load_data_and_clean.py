def main():
    import pandas as pd
    import altair as alt
    import requests
    
    def load_clean_trump_csv(url):
        """
        ### ChatGPT assistance with the creation and debugging of the function to load and clean data ### 
        This function loads the data from the passed URL (specific to this TrumpTweet dataset)
        and cleans the rows for readability with pandas.
        """
        resp = requests.get(url)                              # make HTTP request 
        resp.raise_for_status()                                # checks if request was successful
        lines = resp.text.splitlines()                          # extract response as string as split lines 
        rows = []
        
        for i, line in enumerate(lines, start=1):                  # enumerate lines starting at 1 and return text
            line = line.rstrip("\n\r")                              # remove newline and return characters at end of lines 
            if not line.strip():                                     # skip empty lines 
                continue
            
            parts = line.split(",")                                      # split lines at commas into lists []
            
            if i == 1:                                                      # skip header row
                continue
            
            if len(parts) < 4:                                                 # skip rows with less than 4 columns 
                continue
            
            id_val = parts[0].strip()                                             # extract columns and remove whitespace 
            time_val = parts[1].strip()
            url_val = parts[2].strip()
            tweet_text = ",".join(parts[3:]).strip()                                 # handle rows with commas in text 
            
            rows.append((id_val, time_val, url_val, tweet_text))                        # adds this row as tuple for DF creation 
        
        df = pd.DataFrame(rows, columns=["ID", "Time", "Tweet URL", "Tweet Text"])        # return cleaned data as pandasDF
        return df
    
    url = "https://raw.githubusercontent.com/MarkHershey/CompleteTrumpTweetsArchive/master/data/realDonaldTrump_in_office.csv"
    
    tweets = load_clean_trump_csv(url)                                                # call loac_clean_data function 
    tweets.columns = tweets.columns.str.strip()                                      # clean column names
    tweets["Date & Time"] = pd.to_datetime(tweets["Time"], errors="coerce")         # rename Time column and convert to datetime
    tweets = tweets.drop(columns=["ID", "Tweet URL", "Time"])                      # drop unnecessary columns 
    
    print(tweets.shape)
    print(tweets.head(10))

if __name__ == "__main__":
    main()
    print("Successfully loaded and cleaned data!")