#!/usr/bin/env python3

#80% training data
def test_trainsplit(df,fraction = .8):
    training_restaurants_df = df.sample(frac=fraction, random_state = 42)
    test_restaurants_df = df[~df.isin(training_restaurants_df)].dropna()
    return training_restaurants_df, test_restaurants_df

def get_kfolds(df, folds = 10):
    df_test = []
    df_train = []
    newdf = df
    for i in range(0,folds - 1):
        df_part = newdf.sample(frac= (1/(folds - i)), random_state = 42)
        df_test.append(df_part)
        df_train.append(df[~df.isin(df_part)].dropna())
        newdf = newdf[~newdf.isin(df_part)].dropna()
    df_test.append(newdf) 
    df_train.append(df[~df.isin(newdf)].dropna())
    return df_test,df_train

