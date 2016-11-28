#!/usr/bin/env python3

from math import import nan

import pandas as pd
import numpy as np

DATASET_DIR = '../../dataset/academic'

df_review = pd.read_json('../out/yelp_academic_dataset_review_sentiment.json', lines=True)
df_review = df_review[df_review['sentiment_value'] != 3]    # remove reviews with invalid sentiment_value

# normalize the values to [1, 2]
def normalize(series):
    return ((series - series.min()) / ((series.max() - series.min()) * 1.0)) + 1

def generator_normalized_sentiment_value(df):
    # stars are integers and within [1, 5]
    return c['sentiment_value'] * normalize(c['stars']) * normalize(c['votes'].apply(lambda s: s['useful']))

df_review = df_review.assign(normalized_sentiment_value = lambda df: generator_normalized_sentiment_value(df))
# now the range of normalized_sentiment_value is [-8, 8]

df_business = pd.read_json(DATASET_DIR + '/yelp_academic_dataset_business.json', lines=True)

business_filters = (df_business['categories'].apply(lambda cs: 'Restaurants' in cs)
                    & df_business['open']
                    & df_business['review_count'].apply(lambda rc: rc >= 20))

df_business_restaurants = (df_business[business_filters]
                           .reset_index(drop=True)[['business_id', 'stars', 'hours', 'city', 'attributes']])

# Add a column of 'review_rating' to the business dataframe
review_group_by_business_id = df_review.groupby('business_id')

def get_average_review_rating_of_business(bid):
    group = review_group_by_business_id.get_group(bid)
    if not group.empty:
        # use the mean of the normalized_sentiment_value for the review_rating of a business
        return group['normalized_sentiment_value'].mean()
    else:
        return nan;

df_business_restaurants = df_business_restaurants.assign(
    review_rating = lambda df: df['business_id'].apply(lambda bid: get_average_review_rating_of_business(bid)))

# we found that all businesses have a least one review
# print(len(df_business_restaurants[df_business_restaurants['review_rating'] == nan]))


