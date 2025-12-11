# Trump Twitter Analysis - Makefile
# Author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# Date: 2025-12-08
#
# This Makefile automates the data analysis pipeline for the Trump Twitter
# Sentiment Analysis project.
#
# Usage:
#   make all      - Run the entire pipeline from data download to report
#   make clean    - Remove all generated files
#
# Example with Docker:
#   docker-compose run --rm jupyter-notebook bash -c "cd /home/jovyan/work && make all"

.PHONY: all clean

# Top level target
all: report/trump_twitter_analysis_report.html

# ===================== REPORT =====================
# Final report depends on all figures and tables
report/trump_twitter_analysis_report.html: report/trump_twitter_analysis_report.qmd \
		results/figures/correlation_matrix.png \
		results/figures/anomaly_detection.png \
		results/figures/feature_distributions.png \
		results/figures/tweet_frequency_time_of_day.png \
		results/figures/tweet_frequency_season.png \
		results/figures/sentiment_counts.png \
		results/figures/wordcloud_positive.png \
		results/figures/wordcloud_negative.png \
		results/tables/time_of_day_summary.csv \
		results/tables/season_summary.csv \
		results/tables/sentiment_counts.csv \
		results/tables/top_positive_words.csv \
		results/tables/top_negative_words.csv \
		results/tables/model_metrics.csv
	quarto render report/trump_twitter_analysis_report.qmd --to html

# ===================== WORD CLOUD ANALYSIS =====================
# wordcloud_analysis.py:
#   Input: data/processed/trump_tweets_processed.csv
#   Output: results/figures/wordcloud_positive.png, wordcloud_negative.png
#           results/tables/top_positive_words.csv, top_negative_words.csv, model_metrics.csv
results/figures/wordcloud_positive.png results/figures/wordcloud_negative.png \
results/tables/top_positive_words.csv results/tables/top_negative_words.csv \
results/tables/model_metrics.csv: data/processed/trump_tweets_processed.csv \
		scripts/wordcloud_analysis.py \
		src/sentiment_utils.py \
		src/visualization_utils.py
	python scripts/wordcloud_analysis.py \
		--processed_data=data/processed/trump_tweets_processed.csv \
		--plot_to=results/figures \
		--table_to=results/tables

# ===================== SENTIMENT ANALYSIS =====================
# sentiment_analysis.py:
#   Input: data/processed/trump_tweets_processed.csv
#   Output: data/processed/trump_tweets_with_sentiment.csv
#           results/figures/sentiment_counts.png
#           results/tables/sentiment_counts.csv
results/figures/sentiment_counts.png results/tables/sentiment_counts.csv \
data/processed/trump_tweets_with_sentiment.csv: data/processed/trump_tweets_processed.csv \
		scripts/sentiment_analysis.py
	python scripts/sentiment_analysis.py \
		--processed_data=data/processed/trump_tweets_processed.csv \
		--write_to=data/processed/trump_tweets_with_sentiment.csv \
		--plot_to=results/figures \
		--table_to=results/tables

# ===================== EDA =====================
# eda.py:
#   Input: data/processed/trump_tweets_processed.csv
#   Output: results/figures/tweet_frequency_time_of_day.png, tweet_frequency_season.png
#           results/tables/time_of_day_summary.csv, season_summary.csv
results/figures/tweet_frequency_time_of_day.png results/figures/tweet_frequency_season.png \
results/tables/time_of_day_summary.csv results/tables/season_summary.csv: data/processed/trump_tweets_processed.csv \
		scripts/eda.py \
		src/visualization_utils.py
	python scripts/eda.py \
		--processed_data=data/processed/trump_tweets_processed.csv \
		--plot_to=results/figures \
		--table_to=results/tables

# ===================== PREPROCESSING =====================
# preprocess_validate.py:
#   Input: data/raw/realDonaldTrump_in_office.csv
#   Output: data/processed/trump_tweets_processed.csv
#           results/figures/correlation_matrix.png, anomaly_detection.png, feature_distributions.png
results/figures/correlation_matrix.png results/figures/anomaly_detection.png \
results/figures/feature_distributions.png data/processed/trump_tweets_processed.csv: data/raw/realDonaldTrump_in_office.csv \
		scripts/preprocess_validate.py \
		src/data_utils.py
	python scripts/preprocess_validate.py \
		--raw_data=data/raw/realDonaldTrump_in_office.csv \
		--write_to=data/processed/trump_tweets_processed.csv \
		--plot_to=results/figures

# ===================== DATA DOWNLOAD =====================
# read_trump_tweets.py:
#   Input: URL (hardcoded)
#   Output: data/raw/realDonaldTrump_in_office.csv
data/raw/realDonaldTrump_in_office.csv: scripts/read_trump_tweets.py
	python scripts/read_trump_tweets.py \
		--write_to=data/raw/realDonaldTrump_in_office.csv

# ===================== CLEAN =====================
# Remove all generated files
clean:
	rm -f data/raw/realDonaldTrump_in_office.csv
	rm -f data/processed/trump_tweets_processed.csv
	rm -f data/processed/trump_tweets_with_sentiment.csv
	rm -f results/figures/*.png
	rm -f results/tables/*.csv
	rm -f report/trump_twitter_analysis_report.html
	rm -f report/trump_twitter_analysis_report.pdf