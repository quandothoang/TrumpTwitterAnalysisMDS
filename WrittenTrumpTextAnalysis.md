## Written Trump Twitter Analysis

Group 14: MDS 522 : Members Quan Hoang, Mailys Guegon, Joel Peterson, Li Pu 
Analyzing the realDonaldTrump_in_office.csv 
from https://github.com/MarkHershey/CompleteTrumpTweetsArchive/blob/master/data/realDonaldTrump_in_office.csv

When setting out to analyze this data the main questions of inerest were as follows:

1. Does the time of day/period of the year affect the frequency of the tweets?
2. How many tweets are positive VS negative?
3. What are the most frequent words in the positive and negative tweets?

For the initial EDA and data processing we ended up having to drop around 1/2 of the rows to stop unconventional characters at the end of the tweet string from tripping up pandas. After a visual review we noticed that these problem rows were quite evenly distributed (every 2-3 rows) and that this wouldn't be too much of an issue to get started.

Some somewhat common-sense assumptions we made were that the frequency of tweets would be highest during the daytime and night-time. These assumptions were proven wrong as the numbers show that the most frequent time of day for tweets was the overnight period. 

1. The initial data filtering by time of day visualized in the bar chart shows that the highest frequency of tweets occurs during the overnight period between 12:01am and 8:00am (as mentioned above) with a total of 3706 analyzed tweets. The second most frequent tweet period occurred during the nighttime period between 4:01pm-12:00am with a total of 3653 analyzed tweets, with the time of day resulting in the least frequent amount of tweets actually being the daytime period between 8:01am-4:00pm with a total of 3325 analyzed tweets. Perhaps this is because Trump is most busy during the day (golfing?) and cannot tend to tweets.

2. 