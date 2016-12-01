#!/usr/bin/env python3

from datetime import time
import pandas as pd
import json as json
import numpy as np

def write_df_to_json_file(df, filename):
    df.reset_index().to_json(filename,orient='records')

def write_preprocessed_data(df_business_restaurants):
    write_df_to_json_file(df_business_restaurants,"../out/preprocessed_business_data.json")

# Helper methods for pre-processing

WORKING_TYPES = {
        "WEEKEND_TYPE": "weekend",
        "BREAKFAST_TYPE": "breakfast",
        "LUNCH_TYPE": "lunch",
        "AFTER_LUNCH_TYPE": "after-lunch",
        "DINNER_TYPE": "dinner",
        "NIGHT_TYPE": "night",
    }

breakfast = time(8)
lunch = time(12)
after_lunch = time(15)
dinner = time(18)
night = time(0)

def get_avg_score_of_business(series, bid, col):
    if bid in series.groups:
        group = series.get_group(bid)
        return group[col].mean()
    else:
        return np.nan

def add_evluation_score_to_business(new_col_for_score_df, df_business_restaurants):
    series = new_col_for_score_df[0].groupby('business_id')
    df_business_restaurants[new_col_for_score_df[2]] = df_business_restaurants['business_id'].apply(lambda bid: get_avg_score_of_business(series, bid, new_col_for_score_df[1]))
    return df_business_restaurants

def get_checkin_count(df_checkin, bid):
    row = df_checkin[df_checkin['business_id'] == bid]
    if not row.empty:
        return row.iloc[0]['checkin_count']
    else:
        return np.nan

def score_to_rating(col, df_business_restaurants):
     ## calculate the rating from scores
    NUM_BINS_OF_RATING = 72
    bins = np.linspace(
        df_business_restaurants[col].min(),
        df_business_restaurants[col].max(),
        num=NUM_BINS_OF_RATING)
    df_business_restaurants[col] = df_business_restaurants[col].apply(
        # return the rating which is closeat to the score.
        lambda score: np.argmin(np.abs(bins - score)) + 1)
    return df_business_restaurants

def in_between(start, end, check):
    if start == end: # 24 hours
        return True
    if start < end:
        return start <= check < end
    else: # over midnight e.g., 23:30-04:15
        return start <= check or check < end

TYPE_THRESHOLD = 1
def get_available_working_type(c, wt):
    if c >= TYPE_THRESHOLD:
        return [wt]
    else:
        return []

def spec_hours_to_type(s):
    types = []

    breakfast_count = 0
    lunch_count = 0
    after_lunch_count = 0
    dinner_count = 0
    night_count = 0

    for day in s:

        clo = s[day]['close']
        op = s[day]['open']

        h, m = clo.split(':')
        clo_t = time(int(h), int(m))

        h, m = op.split(':')
        op_t = time(int(h), int(m))

        breakfast_count += int(in_between(op_t, clo_t, breakfast))
        lunch_count += int(in_between(op_t, clo_t, lunch))
        after_lunch_count += int(in_between(op_t, clo_t, after_lunch))
        dinner_count += int(in_between(op_t, clo_t, dinner))
        night_count += int(in_between(op_t, clo_t, night))

        if (day in ['Saturday', 'Sunday']) and (WORKING_TYPES["WEEKEND_TYPE"] not in types):
            types.append(WORKING_TYPES["WEEKEND_TYPE"])

    types += get_available_working_type(breakfast_count, WORKING_TYPES["BREAKFAST_TYPE"])
    types += get_available_working_type(lunch_count, WORKING_TYPES["LUNCH_TYPE"])
    types += get_available_working_type(after_lunch_count, WORKING_TYPES["AFTER_LUNCH_TYPE"])
    types += get_available_working_type(dinner_count, WORKING_TYPES["DINNER_TYPE"])
    types += get_available_working_type(night_count, WORKING_TYPES["NIGHT_TYPE"])

    return join_types(types)

def hours_to_type(s):
    if isinstance(s, str):
        return s

    if s:
        return spec_hours_to_type(s)
    else:
        return join_types(WORKING_TYPES.values())

def join_types(ts):
    # reorder
    ordered_types = []
    for t in WORKING_TYPES.values():
        if t in ts:
            ordered_types.append(t)
    return '_'.join(ordered_types)

def get_preprocessed_data():

    DATASET_DIR = '../../dataset/academic'

    ## read reviews and calculate sentiment scores
    df_review = pd.read_json('../out/yelp_academic_dataset_review_sentiment.json', lines=True)
    df_review = df_review[df_review['sentiment_value'] != 3]    # remove reviews with invalid sentiment_value
    df_review = df_review.assign(sentiment_score =
                                 lambda df: df['sentiment_value'] * df['votes'].apply(lambda s: s['useful'] + 1))
    ## read tips and calculate sentiment scores
    df_tip = pd.read_json('../out/yelp_academic_dataset_tip_sentiment.json', lines=True)
    df_tip = df_tip[df_tip['sentiment_value'] != 3]    # remove reviews with invalid sentiment_value
    df_tip = df_tip.assign(sentiment_score =
            lambda df: df['sentiment_value'] * (df['likes'] + 1))

    ## read business dataset
    df_business = pd.read_json(DATASET_DIR + '/yelp_academic_dataset_business.json', lines=True)
    business_filters = (df_business['review_count'].apply(lambda rc: rc >= 20)
                    & df_business['categories'].apply(lambda cs: 'Restaurants' in cs)
                    & df_business['open'])
    df_business_restaurants = (df_business[business_filters]
                           .reset_index(drop=True)[['business_id', 'stars', 'review_count', 'hours', 'city', 'attributes']])
    # round the stars
    df_business_restaurants['stars'] = df_business_restaurants['stars'].apply(np.round)

    new_col_for_score_dfs = [(df_review, 'sentiment_score', 'review_sentiment_rating'),
                         (df_review, 'stars', 'review_star_rating'),
                         (df_tip, 'sentiment_score', 'tip_rating')]

    for item in new_col_for_score_dfs:
        df_business_restaurants = add_evluation_score_to_business(item, df_business_restaurants)


    ## read checkin count of business and calculate "checkin_rating"
    df_checkin = pd.read_json('../out/business_with_checkin_count.json')

    df_business_restaurants['checkin_rating'] = df_business_restaurants['business_id'].apply(lambda bid: get_checkin_count(df_checkin,bid))

    df_business_restaurants['checkin_rating'] = df_business_restaurants['review_count'] / df_business_restaurants['checkin_rating']


    ## filter out businesses with nan values
    business_ratings_cols = ['review_sentiment_rating',
    #                          'review_star_rating',
                             'tip_rating',]
    #                          'checkin_rating']

    business_nan_filters = np.ones(len(df_business_restaurants), dtype=bool)
    for col in business_ratings_cols:
        business_nan_filters &= np.invert(np.isnan(df_business_restaurants[col]))

    df_business_restaurants = df_business_restaurants[business_nan_filters].reset_index(drop=True)



    score_cols = ['review_sentiment_rating', 'review_star_rating', 'tip_rating', 'checkin_rating']

    # convert score to rating
    for col in score_cols:
        df_business_restaurants = score_to_rating(col, df_business_restaurants)


    ## convert "hours" in business to "working_type"
    df_business_restaurants = df_business_restaurants.assign(working_type=lambda x: x['hours'])


    df_business_restaurants['working_type'] = df_business_restaurants['working_type'].apply(hours_to_type)
    working_type_set = df_business_restaurants['working_type'].unique()

    return df_business_restaurants

def main():
    df = get_preprocessed_data()
    print (df.head())

if __name__ == "__main__":
    main()
