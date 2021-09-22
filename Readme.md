# 🪙Cryptocurrency Closing Price Prediction 
`Summary` → Predict the closing price of crypto currency </br>
`Rank` → 75, appeared as DarthVaderZ (team of 2 people)</br>
`Problem Type` → Supervised Regression </br>
`Technology used` → Deep Learning and Machine learining

__GO TO:__ [`Problem Description`](#ProblemDescription) [`Dataset Overview`](#Dataset-overview)
[`Library Used`](#Library-used) 

## Problem Description
![crypto](https://images.unsplash.com/photo-1629339942248-45d4b10c8c2f?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=1172&q=80)
After the boom and bust of cryptocurrencies’ prices in recent years, cryptocurrencies have been increasingly regarded as an investment asset. Because of their highly volatile nature, there is a need for good predictions on which to base investment decisions. Different existing studies have leveraged machine learning for more accurate cryptocurrency price prediction. We are interested in applying different modeling techniques to samples with different data structures (qualitative and quantitative data) and dimensional features to achieve an optimization in price prediction.

Using the trading time series of a cryptocurrency’s price, in addition to a set of qualitative features (news, social impact, Twitter, Reddit, social media sentiment analysis), we would like to build a model that forecasts a cryptocurrency’s price. In this challenge, we are focusing on the trading time series and how we can optimize currency forecasting. We will predict future cryptocurrency prices.

There are many factors and constraints that can be taken into consideration when increasing or decreasing cryptocurrency prices by the different stakeholders. These factors can be directly seen in newspapers, related websites or social media, for that including these features in the model can add value and predict more accurate cryptocurrency prices.

The target value is the actual price. We have data extracted in an interval of 1h for a period of one year (from 1st of March 2020 to 1st of March 2021). We are interested to predict the values of cryptocurrency prices in specific timestamps that we have in the validation file.

There can be different ways to solve this problem. One can think about using the prices from the different trading platforms as the initial data (the provided dataset) and build forecasting models and/or Neural Networks ones.

The participant that will build a model with the most accurate results will be the winning one.

The challenge does not stop there, as our main goal is to reach and exceed a given threshold (a specified RMSE score) in the final developed model.

The goal is to have predictions that are accurate in a way that it’s mostly similar to the original validation file, to bypass the given threshold evaluation result, and come up with something that is more accurate.

**[More Details](https://zindi.africa/competitions/cryptocurrency-closing-price-prediction)**

## Dataset Overview
| id        | asset_id | open         | high         | low          | volume       | market_cap      | url_shares | unique_url_shares | reddit_posts | reddit_posts_score | reddit_comments | reddit_comments_score | tweets | tweet_spam | tweet_followers | tweet_quotes | tweet_retweets | tweet_replies | tweet_favorites | tweet_sentiment1 | tweet_sentiment2 | tweet_sentiment3 | tweet_sentiment4 | tweet_sentiment5 | tweet_sentiment_impact1 | tweet_sentiment_impact2 | tweet_sentiment_impact3 | tweet_sentiment_impact4 | tweet_sentiment_impact5 | social_score | average_sentiment | news | price_score | social_impact_score | correlation_rank | galaxy_score | volatility | market_cap_rank | percent_change_24h_rank | volume_24h_rank | social_volume_24h_rank | social_score_24h_rank | medium | youtube | social_volume | percent_change_24h  | market_cap_global | close        |
|-----------|----------|--------------|--------------|--------------|--------------|-----------------|------------|-------------------|--------------|--------------------|-----------------|-----------------------|--------|------------|-----------------|--------------|----------------|---------------|-----------------|------------------|------------------|------------------|------------------|------------------|-------------------------|-------------------------|-------------------------|-------------------------|-------------------------|--------------|-------------------|------|-------------|---------------------|------------------|--------------|------------|-----------------|-------------------------|-----------------|------------------------|-----------------------|--------|---------|---------------|---------------------|-------------------|--------------|
| ID_322qz6 | 1        | 9422.849081  | 9428.490628  | 9422.849081  | 713198620.0  | 173763453624.0  | 1689.0     | 817.0             | 55.0         | 105.0              | 61.0            | 271.0                 | 3420.0 | 1671.0     | 11675867.0      | 39.0         | 1343.0         | 448.0         | 2237.0          | 124.0            | 330.0            | 331.0            | 2515.0           | 120.0            | 506133.0                | 1326610.0               | 1159677.0               | 8406185.0               | 281329.0                | 11681999.0   | 3.6               | 69.0 | 2.7         | 3.6                 | 3.3              | 66.0         | 0.0071176  | 1.0             | 606.0                   | 2.0             | 1.0                    | 1.0                   | 2.0    | 5.0     | 4422          | 1.4345161346109587  | 281806567507.0    | 9428.279323  |
| ID_3239o9 | 1        | 7985.359278  | 7992.059917  | 7967.567267  | 400475518.0  | 142694202230.96 | 920.0      | 544.0             | 20.0         | 531.0              | 103.0           | 533.0                 | 1491.0 | 242.0      | 5917814.0       | 195.0        | 1070.0         | 671.0         | 3888.0          | 1.0              | 52.0             | 315.0            | 1100.0           | 23.0             | 1320.0                  | 381117.0                | 1706376.0               | 3754815.0               | 80010.0                 | 5924770.0    | 3.7               | 1.0  | 2.0         | 2.0                 | 1.0              | 43.5         | 0.00941863 | 1.0             |                         |                 |                        |                       |        |         | 2159          | -2.4595073021531104 | 212689713284.66   | 7967.567267  |
| ID_323J9k | 1        | 49202.033778 | 49394.593518 | 49068.057046 | 3017728869.0 | 916697653223.0  | 1446.0     | 975.0             | 72.0         | 1152.0             | 187.0           | 905.0                 | 9346.0 | 4013.0     | 47778746.0      | 104.0        | 2014.0         | 1099.0        | 11476.0         | 331.0            | 923.0            | 864.0            | 6786.0           | 442.0            | 9848462.0               | 5178557.0               | 2145663.0               | 25510267.0              | 5110490.0               | 47796942.0   | 3.7               | 22.0 | 3.1         | 3.0                 | 3.3              | 65.5         | 0.01353005 | 1.0             | 692.0                   | 3.0             | 1.0                    | 1.0                   |        |         | 10602         | 4.942447794031182   | 1530711784042.0   | 49120.738484 |

## Library Used
|Library & Lang|version|
|:-:|:-:|
| `python` | `3.7.11` | 
|numpy|1.19.5|
|pandas|1.1.5|
|matplotlib|3.2.2|
|sklearn|0.24.1|
|keras|2.5.0|
|tqdm|4.59.0|
|seaborn|0.11.1|
|tensorflow|2.5.0|
|livelossplot|0.5.4|

| Enviroment | Used|
| :-------- | :------- |
| `Editor`  |`JupyterLab`| 
| `Runtime type` | `CPU`|
