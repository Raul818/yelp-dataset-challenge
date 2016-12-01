#!/usr/bin/env python3

import preprocessing
import validation
import numpy as np
import pandas as pd



def get_business_bystars(training_restaurants_df, stars_column):
    group_by_stars = training_restaurants_df.groupby(stars_column)
    business_of_stars = {}
    for star in group_by_stars.groups:
        group = group_by_stars.get_group(star)
        business_of_stars[star] = group.assign(working_type=lambda x: x['hours'])
    return business_of_stars

def get_priors(business_of_stars,training_restaurants_df):
    prior_of_stars = {}
    for star in business_of_stars:
        prior_of_stars[star] = len(business_of_stars[star]) * 1.0 / len(training_restaurants_df)
    #print (prior_of_stars)
    #x = list(prior_of_stars.values())
    #normalizing_fact = 1 / np.linalg.norm(x)
    #for k in prior_of_stars:
        #prior_of_stars[k] = prior_of_stars[k] * normalizing_fact
    return prior_of_stars

def get_uniqueattributevalues(atrribute_names, training_restaurants_df, unique_dimension_values):
    for attribute_name in atrribute_names:
        tdf = training_restaurants_df['attributes'].apply(lambda a: extract_value_from_attrs(a, attribute_name))
        unique_dimension_values[attribute_name] = tdf.unique()
        
        
def get_working_type(business_of_stars):
    working_type_set = set()
    for star in business_of_stars:
        business_of_star_df = business_of_stars[star]
        business_of_star_df['working_type'] = business_of_star_df['working_type'].apply(preprocessing.hours_to_type)
        working_type_set |= set(business_of_star_df['working_type'].unique())
    return working_type_set


def get_unique_columnvalues(training_restaurants_df, column_names,unique_dimension_values):
    for column_name in column_names:
        unique_dimension_values[column_name] = training_restaurants_df[column_name].unique()
        
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
    
    working_type = preprocessing.hours_to_type(row_value['hours'])
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


def k_fold_crossvalidation(df_business_restaurants):
    testlist, trainlist = validation.get_kfolds(df_business_restaurants)
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

def main():
    df_business_restaurants = preprocessing.get_preprocessed_data()
    training, test = validation.test_trainsplit(df_business_restaurants)
    dim_freq_map,prior_of_stars = trainNB(training)
    selected_columns = ['review_count','city','review_sentiment_rating','review_star_rating','tip_rating','checkin_rating']
    accuracy,dist,offcount = testNB (test,dim_freq_map, selected_columns,prior_of_stars)
    print ("With review_star_rating, rounded stars -- accuracy,dist,offcount :",accuracy,dist,offcount)
    selected_columns = ['review_count','city','review_sentiment_rating','tip_rating','checkin_rating']
    accuracy,dist,offcount = testNB (test,dim_freq_map, selected_columns,prior_of_stars)
    print ("With out review_star_rating, rounded stars -- accuracy,dist,offcount :",accuracy,dist,offcount)
    print ("k fold cross validation results ",k_fold_crossvalidation(df_business_restaurants))

if __name__ == "__main__":
    main()