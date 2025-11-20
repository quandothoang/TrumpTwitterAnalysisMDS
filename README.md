# Trump Tweet Analysis

Authors: Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu

## About

Here we are analyzing Tweets published by Donald Trump during his first presidency, specifically how the time of day or season affect the frequency of the tweets. Additionally, we did sentiment analysis using a VADER for classification to determine the frequency of positive, negative and neutral tweets. Finally, using a combination of CountVectorizer and Logistic Regression, we used WorlCloud visualization to determine the most frequent positive and negative words.

The dataset we are using is a complete archive of Donald Trump's tweets (also contains deleted tweets) created by Mark Huang. For the purpose of our analysis we are only using the tweets published during his first presidency between 20 Jan 2017 and 08 Jan 2021. The dataset can be found in this repository [CompleteTrumpTweetsArchive](https://github.com/MarkHershey/CompleteTrumpTweetsArchive?tab=readme-ov-file), in the data folder, specifically the file [realDonaldTrump_in_office.csv](https://github.com/MarkHershey/CompleteTrumpTweetsArchive/blob/master/data/realDonaldTrump_in_office.csv). The dataset contains 5 columns (ID, Time, Tweet URL, Tweet Text) and each row represents a tweet.

## Environment and data analysis setup

To set up the necessary packages for this project, run the following from the root of the repository:

``` bash
conda-lock install --name 522_proj conda-lock.yml
```

To open the data analysis, open the TrumpTweetDataAnalysis.ipynb in jupyter lab by running the following:

``` bash
jupyter lab
```

And then open the TrumpTweetDataAnalysis.ipynb. To select the environment we just created, go to the top right corner and under "Select Kernel" choose "Python [conda env:522_proj]".\
To run the data analysis, under "Run" select "Restart Kernel and run all cells".

## Dependencies

-   `conda` (version 25.9.1 or higher)
-   `conda-lock` (version 3.0.4 or higher)
-   `jupyter-lab` (version 4.4.7 or higher)
-   `nb_conda_kernels` (version 2.5.1 or higher)
-   All packages listed in [`environment.yml`](environment.yml)

## License
The project code in the repository is licensed under the MIT license. 
The Trump Tweet Analysis report is licensed under Creative Commons [Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license](https://creativecommons.org/licenses/by-nc-nd/4.0/)
Any use of the work should be properly attributed, including the link to the webpage. 