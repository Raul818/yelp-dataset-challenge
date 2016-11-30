#!/usr/bin/env python3

from datetime import time
import pandas as pd
import numpy as np
import json as json

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
    
    DATASET_DIR = '../'
    
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

df_business_restaurants = get_preprocessed_data()

def get_business_bystars(training_restaurants_df, stars_column):
    group_by_stars = training_restaurants_df.groupby(stars_column)
    business_of_stars = {}
    for star in group_by_stars.groups:
        group = group_by_stars.get_group(star)
        business_of_stars[star] = group.assign(working_type=lambda x: x['hours'])
    return business_of_stars

from sklearn.preprocessing import normalize
def get_priors(business_of_stars,training_restaurants_df):
    prior_of_stars = {}
    for star in business_of_stars:
        prior_of_stars[star] = len(business_of_stars[star]) * 1.0 / len(training_restaurants_df)
    #print (prior_of_stars)
    x = list(prior_of_stars.values())
    normalizing_fact = 1 / np.linalg.norm(x)
    for k in prior_of_stars:
        prior_of_stars[k] = prior_of_stars[k] * normalizing_fact
    return prior_of_stars

def get_uniqueattributevalues(atrribute_names, training_restaurants_df, unique_dimension_values):
    for attribute_name in atrribute_names:
        tdf = training_restaurants_df['attributes'].apply(lambda a: extract_value_from_attrs(a, attribute_name))
        unique_dimension_values[attribute_name] = tdf.unique()
        
        
def get_working_type(business_of_stars):
    working_type_set = set()
    for star in business_of_stars:
        business_of_star_df = business_of_stars[star]
        business_of_star_df['working_type'] = business_of_star_df['working_type'].apply(hours_to_type)
        working_type_set |= set(business_of_star_df['working_type'].unique())
    return working_type_set


def get_unique_columnvalues(training_restaurants_df, column_names,unique_dimension_values):
    for column_name in column_names:
        unique_dimension_values[column_name] = training_restaurants_df['city'].unique()
        
DEFAULT_TYPE = 'default'
def extract_value_from_attrs(attrs, k):
    if k in attrs:
        return attrs[k]
    else:
        return DEFAULT_TYPE
    
def filter_from_attr_val(attr, k, v):
    return k in attr and attr[k] == v

def filter_no_attr(attr, k):
    return k not in attr


def calculate_frequencies(attributes, dimensions, unique_dimension_values, business_of_stars):
    
    dimension_freq_map = {}
    for dimension in (attributes + dimensions):
        dimension_star_map = {}
        dimension_freq_map[dimension] = dimension_star_map
    
    #calculate the frequencies
    for star in business_of_stars:

        business_of_star_df = business_of_stars[star]
        num_business = len(business_of_star_df)

        for dimension in dimensions:
            dim_star_map = dimension_freq_map[dimension]
            dim_freq = {}
            dim_of_business = business_of_star_df.groupby(dimension)
            num_unique_dimensions = len(unique_dimension_values[dimension])
            for grp in dim_of_business.groups:
                # we use the add-one or Laplace smoothing
                dim_freq[grp] = (len(dim_of_business.get_group(grp)) + 1.0) / (num_business + num_unique_dimensions)
            dim_freq[DEFAULT_TYPE] = 1.0 / (num_business + num_unique_dimensions)
            dim_star_map[star] = dim_freq

        for attribute in attributes:
            attr_star_map = dimension_freq_map[attribute]
            attribute_freq = {}
            attr_set = unique_dimension_values[attribute]
            for t in attr_set:
                if t != DEFAULT_TYPE:
                    num = len(business_of_star_df[business_of_star_df['attributes'].apply(lambda attr: filter_from_attr_val(attr, attribute, t))])
                else:
                    num = len(business_of_star_df[business_of_star_df['attributes'].apply(lambda attr: filter_no_attr(attr, attribute))])
                attribute_freq[t] = (num + 1.0) / (num_business + len(attr_set))
            if DEFAULT_TYPE not in  attribute_freq:
                attribute_freq[DEFAULT_TYPE] = 1.0 / (num_business + len(attr_set))
            attr_star_map[star] = attribute_freq
            
    return dimension_freq_map

import numpy as np
import operator

def predict(probs):
    sorted_probs = sorted(probs.items(), key=operator.itemgetter(1))
    return sorted_probs[-1][0]
    
def correctness(stars, estimated_stars):
    return stars == estimated_stars
    
def distance(stars, estimated_stars):
    return abs(stars - estimated_stars)

def calc_probs(row_value, dim_freq_map, selected_columns, prior_of_stars):#hours, city, attrs, sentiment_value, weighted, tip_sentiment, checkin_count):
    #print (row_value)
    probs_of_stars = {}
    
    working_type = hours_to_type(row_value['hours'])
    #print (working_type)
    for star in prior_of_stars:
        prob = np.log(prior_of_stars[star])
        types_freq_of_stars = dim_freq_map[working_type_column]
        #print (types_freq_of_stars)
        prob += np.log(types_freq_of_stars[star].get(working_type, types_freq_of_stars[star]['default']))
        
        for dimension in selected_columns:
            dim_freq_star_map = dim_freq_map[dimension]
            prob += np.log(dim_freq_star_map[star].get(row_value[dimension], dim_freq_star_map[star]['default']))
        
        attrs = row_value['attributes']
        for attribute in atrribute_names:  
            dim_freq_star_map = dim_freq_map[attribute]
            attrcol = extract_value_from_attrs(attrs, attribute)
            #print (attribute, attrcol)
            #print (dim_freq_star_map[star][DEFAULT_TYPE], "\n")
            prob += np.log(dim_freq_star_map[star].get(attrcol, dim_freq_star_map[star][DEFAULT_TYPE]))
        probs_of_stars[star] = prob
    return probs_of_stars


atrribute_names = ['Accepts Credit Cards','Alcohol','Caters','Noise Level','Price Range','Take-out']
column_names = ['review_count','city','review_sentiment_rating','review_star_rating','tip_rating','checkin_rating']
working_type_column = 'working_type'

def trainNB(training_restaurants_df):
    #group by stars 
    business_of_stars = get_business_bystars(training_restaurants_df,'stars')
    #get priors
    prior_of_stars = get_priors(business_of_stars,training_restaurants_df )
    #get unique values for attributes
    unique_dimension_values = {}
    get_uniqueattributevalues(atrribute_names, training_restaurants_df, unique_dimension_values)
     # unique values for columns
    get_unique_columnvalues(training_restaurants_df, column_names,  unique_dimension_values)
    unique_dimension_values[working_type_column] = get_working_type(business_of_stars)
    
    dimension_frequency_map = calculate_frequencies(atrribute_names,column_names+[working_type_column], unique_dimension_values, business_of_stars)
    return dimension_frequency_map, prior_of_stars

def testNB(test_restaurants_df, dim_freq_map, selected_columns, prior_of_stars):
    result = pd.DataFrame()
    result['stars'] = test_restaurants_df['stars']
    result['stars_probs'] = test_restaurants_df.apply(lambda r: calc_probs(r, dim_freq_map, selected_columns, prior_of_stars), axis=1)
    result['estimated_stars'] = result.apply(lambda r: predict(r['stars_probs']), axis=1)
    #write_df_to_json_file(test_restaurants_df[['stars','estimated_weighted_stars','estimated_stars']],"../out/results.json")
    result['correctness'] = result.apply(lambda r: correctness(r['stars'], r['estimated_stars']), axis=1)
    corrects = len(result[result['correctness'] == True])
    result['distance'] = result.apply(lambda r: distance(r['stars'], r['estimated_stars']), axis=1)
    result['diff'] = result.apply(lambda r: r['stars'] - r['estimated_stars'], axis=1)
    result_t = result[result['diff'].apply(lambda x: abs(x) >= 0.5)]
    accuracy =  corrects * 1.0 / len(result)
    avg_dist = result['distance'].mean()
    off_by_morethan_halfstar = len(result_t)
    return accuracy,avg_dist,off_by_morethan_halfstar

#80% training data
def test_trainsplit(df,fraction = .8):
    training_restaurants_df = df.sample(frac=fraction, random_state = 42)
    test_restaurants_df = df[~df.isin(training_restaurants_df)].dropna()
    return training_restaurants_df, test_restaurants_df

training, test = test_trainsplit(df_business_restaurants)
dim_freq_map,prior_of_stars = trainNB(training)
selected_columns = ['review_count','city','review_sentiment_rating','review_star_rating','tip_rating','checkin_rating']
accuracy,dist,offcount = testNB (test,dim_freq_map, selected_columns,prior_of_stars)
print ("With review -- accuracy,dist,offcount :",accuracy,dist,offcount)
selected_columns = ['review_count','city','review_sentiment_rating','tip_rating','checkin_rating']
accuracy,dist,offcount = testNB (test,dim_freq_map, selected_columns,prior_of_stars)
print ("weighted -- accuracy,dist,offcount :",accuracy,dist,offcount)

def get_kfolds(df, folds = 10):
    df_test = []
    df_train = []
    newdf = df
    for i in range(0,9):
        df_part = newdf.sample(frac= (1/(folds - i)), random_state = 42)
        df_test.append(df_part)
        df_train.append(df[~df.isin(df_part)].dropna())
        newdf = newdf[~newdf.isin(df_part)].dropna()
    df_test.append(newdf) 
    df_train.append(df[~df.isin(newdf)].dropna())
    return df_test,df_train

def k_fold_crossvalidation():
    testlist, trainlist = get_kfolds(df_business_restaurants)
    accuracy_list = []
    dist_list = []
    offcount_list = []
    for i in range(0, 10):
        test = testlist[i]
        train = trainlist[i]
        dim_freq_map,prior_of_stars = trainNB(train)
        selected_columns = ['city','review_sentiment_rating','review_star_rating','tip_rating','checkin_rating']
        accuracy,dist,offcount = testNB (test,dim_freq_map, selected_columns,prior_of_stars)
        accuracy_list.append(accuracy)
        dist_list.append(dist)
        offcount_list.append(offcount)
    return np.mean(np.asarray(accuracy_list)), np.mean(np.asarray(dist_list)), np.mean(np.asarray(offcount_list))

