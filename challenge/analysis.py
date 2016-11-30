#!/usr/bin/env python3

from datetime import time

import pandas as pd
import numpy as np
import operator

DATASET_DIR = '../../dataset/academic'

## read reviews and calculate sentiment scores
df_review = pd.read_json('../out/yelp_academic_dataset_review_sentiment.json', lines=True)
df_review = df_review[df_review['sentiment_value'] != 3]    # remove reviews with invalid sentiment_value
df_review = df_review.assign(
    sentiment_score =
        lambda df: df['sentiment_value'] * df['votes'].apply(lambda s: s['useful'] + 1))


## read tips and calculate sentiment scores
df_tip = pd.read_json('../out/yelp_academic_dataset_tip_sentiment.json', lines=True)
df_tip = df_tip[df_tip['sentiment_value'] != 3]    # remove reviews with invalid sentiment_value
df_tip = df_tip.assign(
    sentiment_score =
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


## merge information from review and tip datasets to business
def get_avg_score_of_business(series, bid, col):
    if bid in series.groups:
        group = series.get_group(bid)
        return group[col].mean()
    else:
        return np.nan

new_col_for_score_dfs = [(df_review, 'sentiment_score', 'review_sentiment_rating'),
                         (df_review, 'stars', 'review_star_rating'),
                         (df_tip, 'sentiment_score', 'tip_rating')]

def add_evluation_score_to_business(new_col_for_score_df):
    series = new_col_for_score_df[0].groupby('business_id')
    global df_business_restaurants
    df_business_restaurants[new_col_for_score_df[2]] = df_business_restaurants['business_id'].apply(
        lambda bid: get_avg_score_of_business(series, bid, new_col_for_score_df[1]))

for item in new_col_for_score_dfs:
    add_evluation_score_to_business(item)


## read checkin count of business and calculate "checkin_rating"
df_checkin = pd.read_json('../out/business_with_checkin_count.json')

def get_checkin_count(bid):
    row = df_checkin[df_checkin['business_id'] == bid]
    if not row.empty:
        return row.iloc[0]['checkin_count']
    else:
        return np.nan

df_business_restaurants['checkin_rating'] = df_business_restaurants['business_id'].apply(
    lambda bid: get_checkin_count(bid))

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


## calculate the rating from scores
NUM_BINS_OF_RATING = 72

def score_to_rating(col):
    bins = np.linspace(
        df_business_restaurants[col].min(),
        df_business_restaurants[col].max(),
        num=NUM_BINS_OF_RATING)

    df_business_restaurants[col] = df_business_restaurants[col].apply(
        # return the rating which is closeat to the score.
        lambda score: np.argmin(np.abs(bins - score)) + 1)

score_cols = ['review_sentiment_rating', 'review_star_rating', 'tip_rating', 'checkin_rating']

# convert score to rating
for col in score_cols:
    score_to_rating(col)


## convert "hours" in business to "working_type"
df_business_restaurants = df_business_restaurants.assign(working_type=lambda x: x['hours'])

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


df_business_restaurants['working_type'] = df_business_restaurants['working_type'].apply(hours_to_type)
working_type_set = df_business_restaurants['working_type'].unique()

DEFAULT_TYPE = 'default'
def extract_value_from_attrs(attrs, k):
    if k in attrs:
        return attrs[k]
    else:
        return DEFAULT_TYPE

def filter_from_attr_val(attrs, k, v):
    return k in attrs and attrs[k] == v

def filter_no_attr(attrs, k):
    return k not in attrs

attr_types_set = {
    'Accepts Credit Cards': [],
    'Alcohol': [],
    'Caters': [],
    'Noise Level': [],
    'Price Range': [],
    'Take-out': []
}

for t in attr_types_set:
    tdf = df_business_restaurants['attributes'].apply(lambda a: extract_value_from_attrs(a, t))
    attr_types_set[t] = tdf.unique().tolist()


## training method
ATTR_MARK = '|attr'

def train(training_dataset):
    business_restaurants_group_by_stars = training_dataset.groupby('stars')

    business_restaurants_of_stars = {}
    for star in business_restaurants_group_by_stars.groups:
        business_restaurants_of_stars[star] = business_restaurants_group_by_stars.get_group(star)

    prior_of_stars = {}
    for star in business_restaurants_of_stars:
        prior_of_stars[star] = len(business_restaurants_of_stars[star]) * 1.0 / len(training_dataset)

    num_working_types = len(working_type_set)
    num_cities = len(training_dataset['city'].unique())

    features_of_stars = {}

    def set_value_in_features_dict(k, star, v):
        if k not in features_of_stars:
            features_of_stars[k] = {}
        features_of_stars[k][star] = v

    for star in business_restaurants_of_stars:
        business_restaurants_of_star_df = business_restaurants_of_stars[star]
        num_business = len(business_restaurants_of_star_df)

        # count frequency of different type
        types_freq = {}

        working_type_of_business = business_restaurants_of_star_df.groupby('working_type')
        for wt in working_type_of_business.groups:
            # we use the add-one or Laplace smoothing
            types_freq[wt] = (len(working_type_of_business.get_group(wt)) + 1.0) / (num_business + num_working_types)

        # this value is for working type not present in this star level
        types_freq[DEFAULT_TYPE] = 1.0 / (num_business + num_working_types)

        set_value_in_features_dict('working_type', star, types_freq)


        # count frequency of different city
        city_freq = {}

        # Now group them by city
        city_of_business = business_restaurants_of_star_df.groupby('city')
        for city in city_of_business.groups:
            # we use the add-one or Laplace smoothing
            city_freq[city] = (len(city_of_business.get_group(city)) + 1.0) / (num_business + num_cities)

        # this value is for cities not in the a specify "group of star",
        # e.g. city "glendale" is not in group of star 1
        city_freq[DEFAULT_TYPE] = 1.0 / (num_business + num_cities)

        set_value_in_features_dict('city', star, city_freq)


        # count frequency of attrs
        for a in attr_types_set:
            attr_freq = {}
            for t in attr_types_set[a]:
                if t != DEFAULT_TYPE:
                    num = len(business_restaurants_of_star_df[business_restaurants_of_star_df['attributes'].apply(lambda attrs: filter_from_attr_val(attrs, a, t))])
                else:
                    num = len(business_restaurants_of_star_df[business_restaurants_of_star_df['attributes'].apply(lambda attrs: filter_no_attr(attrs, a))])

                attr_freq[t] = (num + 1.0) / (num_business + len(attr_types_set[a]))

            set_value_in_features_dict(a + ATTR_MARK, star, attr_freq)


        # count frequency of ratings
        for rt_col in business_ratings_cols:
            rating_freq = {}

            rating_of_business = business_restaurants_of_star_df.groupby(rt_col)
            for rt in rating_of_business.groups:
                rating_freq[rt] = (len(rating_of_business.get_group(rt)) + 1.0) / (num_business + NUM_BINS_OF_RATING)

                rating_freq[DEFAULT_TYPE] = 1.0 / (num_business + NUM_BINS_OF_RATING)

            set_value_in_features_dict(rt_col, star, rating_freq)

    return prior_of_stars, features_of_stars


## testing related method
def calc_probs(r, prior_of_stars, features_of_stars):
    probs_of_stars = {}

    for star in prior_of_stars:
        prob = np.log(prior_of_stars[star])

        for f in features_of_stars:
            if not f.endswith(ATTR_MARK):
                prob += np.log(features_of_stars[f][star].get(r[f], features_of_stars[f][star][DEFAULT_TYPE]))
            else:
                prob += np.log(features_of_stars[f][star].get(
                        extract_value_from_attrs(r['attributes'], f[:-len(ATTR_MARK)]), features_of_stars[f][star].get(DEFAULT_TYPE, np.nan)))

        probs_of_stars[star] = prob

    return probs_of_stars

def predict(stars, probs):
    sorted_probs = sorted(probs.items(), key=operator.itemgetter(1))
    return sorted_probs[-1][0]

def correctness(stars, estimated_stars):
    return stars == estimated_stars

def distance(stars, estimated_stars):
    return abs(stars - estimated_stars)

def test(test_dataset, prior_of_stars, features_of_stars):
    test_dataset['stars_probs'] = test_dataset.apply(lambda r: calc_probs(r, prior_of_stars, features_of_stars), axis=1)

    test_dataset['estimated_stars'] = test_dataset.apply(lambda r: predict(r['stars'], r['stars_probs']), axis=1)
    test_dataset['correctness'] = test_dataset.apply(lambda r: correctness(r['stars'], r['estimated_stars']), axis=1)
    test_dataset['distance'] = test_dataset.apply(lambda r: distance(r['stars'], r['estimated_stars']), axis=1)


    corrects = len(test_dataset[test_dataset['correctness'] == True])
    print('accuracy is ' + str(corrects * 1.0 / len(test_dataset)))

    print('average distance is ' + str(test_dataset['distance'].mean()))


## Split dataset and start train
df_training_business_restaurants = df_business_restaurants.sample(frac=0.8)
prior_of_stars, features_of_stars = train(df_training_business_restaurants)

## Evaluate on the test dataset
df_test_business_restaurants = df_business_restaurants[~df_business_restaurants.isin(df_training_business_restaurants)].dropna()
test(df_test_business_restaurants, prior_of_stars, features_of_stars)
