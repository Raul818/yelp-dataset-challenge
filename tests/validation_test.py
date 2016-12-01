import unittest
import pandas as pd

import validation

# Test for methods in challenge/validation.py
class TestValidationMethods(unittest.TestCase):
    
    def setUp(self):
        # dummy dataframe for testing
        self.dummy_df = pd.DataFrame()
        self.dummy_df['col1'] = ['1','2','3','4','5','6','7','8','9','10']
        self.dummy_df['col2'] = ['A','B','C','D','E','F','G','H','I','J']
    
    # test method for test_trainsplit
    def test_test_trainsplit(self):
        train, test = validation.test_trainsplit(self.dummy_df)
        self.assertEqual( len(train) + len(test), 10)
        self.assertEqual( len(train) , 8)
        self.assertEqual( len(test) , 2)
        self.assertTrue (pd.concat([train, test]).sort_index().equals(self.dummy_df))
    
    # test method for get_kfolds
    def test_get_kfolds(self):
        test_list , train_list = validation.get_kfolds(self.dummy_df, 5)
        self.assertEqual( len(test_list) , 5)
        self.assertEqual( len(train_list) , 5)
        for i in range(0,5):
            self.assertEqual( len(train_list[i]) + len(test_list[i]), 10)
            self.assertTrue (pd.concat([train_list[i], test_list[i]]).sort_index().equals(self.dummy_df))
    
    def tearDown(self):
        self.dummy_df = None
        
       
if __name__ == '__main__':
    unittest.main()
    