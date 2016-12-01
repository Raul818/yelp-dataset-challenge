import unittest
import pandas as pd


import analysis


# Test for methods in challenge/validation.py
class TestAnalysisMethods(unittest.TestCase):
    
    def setUp(self):
        # dummy dataframe for testing
        self.unittest_df = pd.read_json('out/unittest.json')
        self.business_of_stars = analysis.get_business_bystars(self.unittest_df, 'stars')
        self.priors = analysis.get_priors(self.business_of_stars,self.unittest_df)
        self.atrribute_names = ['Accepts Credit Cards','Alcohol','Caters','Noise Level','Price Range','Take-out']
        self.unique_dimension_values = {}
        analysis.get_uniqueattributevalues(self.atrribute_names, self.unittest_df, self.unique_dimension_values)
        self.working_type = analysis.get_working_type(self.business_of_stars)
        self.column_names = ['review_count','city','review_sentiment_rating','review_star_rating','tip_rating','checkin_rating']
        analysis.get_unique_columnvalues(self.unittest_df, self.column_names,  self.unique_dimension_values)
        self.working_type_column = 'working_type'
        self.unique_dimension_values[self.working_type_column] = self.working_type
        self.dimension_frequency_map = analysis.calculate_frequencies(self.atrribute_names,self.column_names+[self.working_type_column], self.unique_dimension_values, self.business_of_stars)
        
    def test_get_business_bystars(self):
        # Only two stars in the test dataset
        business_of_stars = self.business_of_stars
        self.assertEqual( 2 , len(business_of_stars))
        self.assertEqual(4,len(business_of_stars[2]))
        self.assertEqual(6,len(business_of_stars[4]))
        
    def test_get_priors(self):
        #print (self.priors)
        self.assertEqual(.4,self.priors[2])
        self.assertEqual(.6,self.priors[4])
        
    def test_get_uniqueattributevalues(self):
        #Check if the values match expected unique values
        values = self.unique_dimension_values['Accepts Credit Cards']
        self.assertEqual(2,len(values))
        self.assertTrue(True in values and False in values)
        values = self.unique_dimension_values['Take-out']
        self.assertEqual(1,len(values))
        self.assertTrue(True in values)
        values = self.unique_dimension_values['Alcohol']
        self.assertEqual(2,len(values))
        self.assertTrue('full_bar' in values and 'none' in values)
        values = self.unique_dimension_values['Noise Level']
        self.assertEqual(3,len(values))
        self.assertTrue('average' in values and 'loud' in values and 'very_loud' in values)
        values = self.unique_dimension_values['Caters']
        self.assertEqual(3,len(values))
        self.assertTrue(True in values and False in values and 'default' in values)
        values = self.unique_dimension_values['Caters']
        self.assertEqual(3,len(values))
        self.assertTrue(True in values and False in values and 'default' in values)
        values = self.unique_dimension_values['Price Range']
        self.assertEqual(2,len(values))
        self.assertTrue(2 in values and 1 in values)
    
    def test_get_working_type(self):
        self.assertEqual(4,len(self.working_type))
        expectedset = {'after-lunch_dinner_lunch_weekend', 'breakfast_lunch_weekend', 'after-lunch_dinner_lunch_night_weekend', 'after-lunch_breakfast_dinner_lunch_night_weekend'}
        for val in expectedset:
            self.assertTrue(val in self.working_type)
        
    def test_get_unique_columnvalues(self):
        values = self.unique_dimension_values['city']
        self.assertTrue('Braddock' in values and 'Carnegie' in values and 'Munhall' in values and 'Homestead' in values and 'West Homestead' in values)
        values = self.unique_dimension_values['review_count']
        self.assertEqual(9,len(values))
     
    def test_calculate_frequencies(self):
        self.unique_dimension_values[self.working_type_column] = analysis.get_working_type(self.business_of_stars)
        dimension_frequency_map = analysis.calculate_frequencies(self.atrribute_names,self.column_names+[self.working_type_column], self.unique_dimension_values, self.business_of_stars)
        #expected frequencies for 13 features
        self.assertEqual(13, len(dimension_frequency_map))
        #testing frequency of one easily calculated feature
        values = dimension_frequency_map['Alcohol'][4]
        self.assertTrue(values['full_bar'] == 0.75 and values['none'] == 0.25 )
        
if __name__ == '__main__':
    unittest.main()
    