# Trump Tweet Analysis

Authors: Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu

## About

Here we are analyzing Tweets published by Donald Trump during his first presidency, specifically how the time of day or season affect the frequency of the tweets. Additionally, we did sentiment analysis using a VADER for classification to determine the frequency of positive, negative and neutral tweets. Finally, using a combination of CountVectorizer and Logistic Regression, we used WorldCloud visualization to determine the most frequent positive and negative words.

The dataset we are using is a complete archive of Donald Trump's tweets (also contains deleted tweets) created by Mark Huang. For the purpose of our analysis we are only using the tweets published during his first presidency between 20 Jan 2017 and 08 Jan 2021. The dataset can be found in this repository [CompleteTrumpTweetsArchive](https://github.com/MarkHershey/CompleteTrumpTweetsArchive?tab=readme-ov-file), in the data folder, specifically the file [realDonaldTrump_in_office.csv](https://github.com/MarkHershey/CompleteTrumpTweetsArchive/blob/master/data/realDonaldTrump_in_office.csv). The dataset contains 5 columns (ID, Time, Tweet URL, Tweet Text) and each row represents a tweet.

## Dependencies

-   [Docker](https://www.docker.com/)
-   [VS Code](https://code.visualstudio.com/download)
-   [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## Usage

(The usage structure is inspired by Tiffany Timbers' Usage Readme section [breast-cancer-predictor repository](https://github.com/ttimbers/breast-cancer-predictor))

### Setup

Clone this Github repository by running :

```         
git clone https://github.com/quandothoang/TrumpTwitterAnalysisMDS.git
```

### Running the analysis

1.  Open Docker Desktop

2.  Run the following from the root of the repository to launch the container:

```         
docker compose up
```

3.  In the terminal, towards the bottom of the output, there should be a URL that starts with `http://127.0.0.1:8888/lab?token=`. Change the port `8888` to `8787` in the URL, then copy and paste it into your browser.

4.  To run the analysis, open `work/TrumpTweetDataAnalysis.ipynb` in Jupyter Lab and under "Run" select "Restart Kernel and run all cells".

### Clean up

To shut down the container, type `Ctrl` + `C` in the terminal where you launched the container, hten type `docker compose rm`.

## License

The project code in the repository is licensed under the MIT license. The Trump Tweet Analysis report is licensed under Creative Commons [Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license](https://creativecommons.org/licenses/by-nc-nd/4.0/) Any use of the work should be properly attributed, including the link to the webpage.
