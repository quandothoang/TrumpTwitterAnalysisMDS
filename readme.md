# Trump Tweet Analysis

Authors: Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu

## About

In this report we will be analyzing Tweets published by Donald Trump during his first presidency, focusing on two different aspects : frequency and sentiment. 
We started by considering how the time of day and season affects the frequency of the tweets. 
Our research reveals that there is a variation in the frequency depending on time of day and 
that there is a 77.14% increase in the number of tweets published from the daytime to the evening. 
We also found that changing the season also had an impact on the frequency, with there being a 53.87% increase in the number of tweets 
from the season where he posts the least (Winter) to the season where he posts the most (Summer). 
Additionally, we performed sentiment analysis using the VADER lexicon to classify the tweets and determine the frequency of positive, negative and neutral tweets. 
We found that around 51.29% of the tweets were classified as positive and 29.62% as negative. 
Using the positive and negative labels, we will train a Logistic Regression model to determine the most frequently used words in the positive and negative tweets. 
We found that the positive tweets most often contained words of praise and common phrases used by Trump, as well as his own name, 
while the negative tweets most often used words to refer to opposing political parties and leaders. 



The dataset we are using is a complete archive of Donald Trump's tweets (also contains deleted tweets) created by Mark Huang. 
For the purpose of our analysis we are only using the tweets published during his first presidency between 20 Jan 2017 and 08 Jan 2021.
The dataset can be found in this repository [CompleteTrumpTweetsArchive](https://github.com/MarkHershey/CompleteTrumpTweetsArchive?tab=readme-ov-file), in the data folder, 
specifically the file [realDonaldTrump_in_office.csv](https://github.com/MarkHershey/CompleteTrumpTweetsArchive/blob/master/data/realDonaldTrump_in_office.csv). 
The dataset contains 5 columns (ID, Time, Tweet URL, Tweet Text) and each row represents a tweet.

## Summary of Findings

Our analysis reveals that Trump tweeted most frequently during **evening hours** (77% more than daytime) and during **summer months**. Sentiment analysis showed that 51% of tweets were positive, with words like "great" and "MAGA" appearing frequently, while negative tweets often contained words like "fake" and "corrupt".


## Dependencies
See [environment.yml](environment.yml) for the complete list. Key packages include:
- 	Python 3.11
- 	pandas, scikit-learn, altair
- 	vaderSentiment (for sentiment analysis)
- 	quarto (for report rendering)
-   [Docker](https://www.docker.com/)
-   [VS Code](https://code.visualstudio.com/download)
-   [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

**Note:** For Docker users, use `conda-linux-64.lock`. Other lock files are for different platforms.


## Usage

(The usage structure is inspired by Tiffany Timbers' Usage Readme section [breast-cancer-predictor repository](https://github.com/ttimbers/breast-cancer-predictor))

### Setup

Clone this Github repository by running :

```         
git clone https://github.com/quandothoang/TrumpTwitterAnalysisMDS.git
```

### Running the analysis

1.  Open Docker Desktop

2.  Run the following command from the root of the repository to launch the container:

```         
docker compose up
```

3.  In the terminal, towards the bottom of the output, there should be a URL that starts with `http://127.0.0.1:8888/lab?token=`. 
Change the port `8888` to `8787` in the URL, then copy and paste it into your browser (the URL should now start with :`http://127.0.0.1:8787/lab?token=`).

4.  To run the analysis, open a terminal in the notebook (the first line should look like this: `(base) jovyan@ce7534bf3379:~$`) and run the following commands: 
```
cd work/
python scripts/read_trump_tweets.py
python scripts/preprocess_validate.py
python scripts/eda.py
python scripts/sentiment_analysis.py
python scripts/wordcloud_analysis.py 
quarto render report/trump_twitter_analysis_report.qmd
```
(All arguments for the scripts are optional, for more information look at the scripts' docstrings)

5. The rendered report can be found by running in the command line :
For the pdf report:
```
cd report/trump_twitter_analysis_report.pdf
```
or for the html report :
```
cd report/trump_twitter_analysis_report.html
```
and for the quarto report :
```
cd report/trump_twitter_analysis_report.qmd
```

### Running the analysis with Make (Recommended)

Alternatively, you can use Make to run the entire pipeline automatically:
```bash
cd work
make clean   # Remove all generated files
make all     # Run the entire pipeline
```

This will execute the full analysis pipeline: download raw data → preprocess and validate → generate EDA visualizations → perform sentiment analysis → create word clouds → render the final report.

To force rebuild everything (ignore existing files):
```bash
make -B all
```

### Clean up

To shut down the container, type `Ctrl` + `C` in the terminal where you launched the container, then type `docker compose rm`.

## Developer notes
(The developer notes section structure is taken from Tiffany Timbers' Developer Notes Readme section [breast-cancer-predictor repository](https://github.com/ttimbers/breast-cancer-predictor))

### Developer dependencies

- `conda` (version 23.9.0 or higher)
- `conda-lock` (version 2.5.7 or higher)
### Adding a new dependency

- Add the dependency to the environment.yml file on a new branch.

- Run conda-lock -k explicit --file environment.yml -p linux-64 to update the conda-linux-64.lock file.

- Re-build the Docker image locally to ensure it builds and runs properly.

- Push the changes to GitHub. A new Docker image will be built and pushed to Docker Hub automatically. It will be tagged with the SHA for the commit that changed the file.

- Update the docker-compose.yml file on your branch to use the new container image (make sure to update the tag specifically).

- Send a pull request to merge the changes into the main branch.

## License

The project code in the repository is licensed under the MIT license. The Trump Tweet Analysis report is licensed under Creative Commons [Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license](https://creativecommons.org/licenses/by-nc-nd/4.0/) Any use of the work should be properly attributed, including the link to the webpage.

## Attribution
ChatGPT was used in this project for troubleshooting and refining parts of the code.

## References

Gottfried, J., Park, E., & Nolan, H. (n.d.). Americans’ Social Media Use 2025 Growing shares of U.S. adults say they are using Instagram, TikTok, WhatsApp and Reddit, but YouTube still rises to the top for media or other inquiries. Retrieved November 21, 2025, from <https://www.pewresearch.org/wp-content/uploads/sites/20/2025/11/PI_2025.11.20_Social-Media-Use_REPORT.pdf>

McCarthy, N. (2021, January 11). Infographic: End Of The Road For Trump’s Twitter Account. Statista Daily Data; Statista. <https://www.statista.com/chart/19561/total-number-of-tweets-from-donald-trump/?srsltid=AfmBOorYyvrCIBJxWCwAxW5yYEl6cPXzdhu-oMfRfAPfoXrcdpIEA3fy>

Mythili Sampathkumar. (2018, January 17). The tweets that have defined Donald Trump’s presidency \| The Independent. The Independent. <https://www.independent.co.uk/news/world/americas/us-politics/donald-trump-twitter-president-first-year-a8163791.html>

Shear, M. D., Haberman, M., Confessore, N., Yourish, K., Buchanan, L., & Collins, K. (2019, November 2). How Trump Reshaped the Presidency in Over 11,000 Tweets. The New York Times. <https://www.nytimes.com/interactive/2019/11/02/us/politics/trump-twitter-presidency.html>
