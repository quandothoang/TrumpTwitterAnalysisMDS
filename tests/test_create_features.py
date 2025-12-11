# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-08

"""
Tests for the create_features() function in data_utils module.

The create_features() is used to feature engineer numerical and categorical features 
from the original Date & Time and Tweet Text features.

Test categories:
1. Normal cases - Normal dataframe input
2. Edge cases - Empty string as tweet and tweet with only punctuation 
3. Error cases - Invalid inputs

Run tests with: pytest tests/test_create_features.py -v
"""

import os
import pytest
import pandas as pd
import sys
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from src.data_utils import create_features, avg_word_length, punctuation_count

# ----------------------------------------------------------------------------- #
# Normal case test data
# ----------------------------------------------------------------------------- #
@pytest.fixture
def normal_test_case():
    return pd.DataFrame({
    'Date & Time' : [pd.Timestamp(2021,1,6,18,13,14),pd.Timestamp(2014,4,5,13,14,17), pd.Timestamp(2025,8,2,1,49,5)],
    'Tweet Text' : ['I am asking for everyone at the U.S. Capitol to remain peaceful. No violence! Remember, WE are the Party of Law & Order – respect the Law and our great men and women in Blue. Thank you!', 
                    "RT @KLoeffler: It's lunchtime. Have you voted yet? If you haven't — GO VOTE and bring 10 people know! If you have — call your family, friends and neighbors to make sure they have too! #gapol #gasen",
                    "I am asking all America First Patriots in Tennessee’s 7th Congressional District to please GET OUT AND VOTE for a phenomenal Candidate and MAGA Warrior, Matt Van Epps! You can win this Election for Matt, who has my Complete and Total Endorsement. HE WILL BE A GREAT CONGRESSMAN"]
})

@pytest.fixture
def normal_test_output(normal_test_case):
    return pd.concat([normal_test_case.copy().reset_index(), pd.DataFrame({
    'length' : [len(normal_test_case.loc[0,'Tweet Text']), 
                len(normal_test_case.loc[1,'Tweet Text']),
                len(normal_test_case.loc[2,'Tweet Text'])],
    'hour'  :  [normal_test_case.loc[0,'Date & Time'].hour, 
                normal_test_case.loc[1,'Date & Time'].hour,
                normal_test_case.loc[2,'Date & Time'].hour],
    'weekday' : [normal_test_case.loc[0,'Date & Time'].weekday(),
                 normal_test_case.loc[1,'Date & Time'].weekday(), 
                 normal_test_case.loc[2,'Date & Time'].weekday()],
    'year' : [normal_test_case.loc[0,'Date & Time'].year,
              normal_test_case.loc[1,'Date & Time'].year, 
              normal_test_case.loc[2,'Date & Time'].year],
    'month' : [normal_test_case.loc[0,'Date & Time'].month,
               normal_test_case.loc[1,'Date & Time'].month, 
               normal_test_case.loc[2,'Date & Time'].month],
    'day' : [normal_test_case.loc[0,'Date & Time'].day,
             normal_test_case.loc[1,'Date & Time'].day, 
             normal_test_case.loc[2,'Date & Time'].day],
    'season' : ['winter', 
                'spring', 
                'summer'],
    'time_of_day' : ['evening', 
                     'daytime', 
                     'overnight'],
    'avg_word_length' : normal_test_case["Tweet Text"].apply(avg_word_length).tolist(),
    'word_count' : normal_test_case["Tweet Text"].apply(lambda x: len(re.sub(r"[^A-Za-z\s]", "", x).split())).tolist(),
    'punctuation_count' : normal_test_case["Tweet Text"].apply(punctuation_count).tolist()
})], axis=1)


# ----------------------------------------------------------------------------- #
# Edge case test data
# ----------------------------------------------------------------------------- #

# Case 1 : Tweet is an empty string
# Expected behaviour : avg_word_length, word_count, punctuation_count are all 0
@pytest.fixture
def edge_case_1():
    return pd.DataFrame({
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14)],
    'Tweet Text' : ['']
})



# Case 2 : Tweet contains only punctuation
# Expected behaviour : avg_word_length and word_count are 0
@pytest.fixture
def edge_case_2(): 
    return pd.DataFrame({
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14)],
    'Tweet Text' : ['...///...^']
})


# ----------------------------------------------------------------------------- #
# Error handling test data 
# ----------------------------------------------------------------------------- #

# Case 1 : Input is not a dataframe
# Expected behaviour : Should raise TypeError
@pytest.fixture
def error_case_1(): 
    return {
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14)],
    'Tweet Text' : ['I am a concerned citizen!']
}


# Case 2 : The columns in the input dataframe are not correct
# Expected behaviour : Should raise a ValueError
@pytest.fixture
def error_case_2(): 
    return pd.DataFrame({
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14)],
    'Tweet' : ['I am a concerned citizen!']
})


# Case 3 : The Date & Time column does not have the right type
# Expected behaviour : Should raise a TypeError
@pytest.fixture
def error_case_3(): 
    return pd.DataFrame({
    'Date & Time'  : ['2012/12/24'],
    'Tweet Text' : ['I am a concerned citizen!']
})

# Case 4 : The Tweet Text column does not have the right type 
# Expected behaviour : Should raise a ValueError
@pytest.fixture
def error_case_4():  
    return pd.DataFrame({
    'Date & Time'  : [pd.Timestamp(2012,12,12,12,22,14)],
    'Tweet Text' : [1]
})


# ----------------------------------------------------------------------------- #
# create_features() – normal cases                                              #
# ----------------------------------------------------------------------------- #

def test_create_features_normal(normal_test_case,normal_test_output):
    """ Tests to ensure create_features function correctly adds features."""
    results = create_features(normal_test_case)

    # Verify return type 
    assert isinstance(results, pd.DataFrame)

    # Verify that the output dataframe is as expected
    pd.testing.assert_frame_equal(normal_test_output, results, check_dtype = False)



# ----------------------------------------------------------------------------- #
# create_features() – edge cases                                                #
# ----------------------------------------------------------------------------- #

def test_create_features_edge(edge_case_1,edge_case_2):
    """ Tests to ensure create_features function works correctly when the string is empty or when there is only punctuation."""

    # Test empty tweet - should return 0 for length, avg_word_length, word_count, punctuation_count
    result_empty = create_features(edge_case_1)
    assert result_empty.iloc[0, 11:].all() == 0 
    
    # Test punctuation tweet - should return 0 for word_count and avg_word_length
    result_punct = create_features(edge_case_2)
    assert result_punct.iloc[0, 11:13].all() == 0 


# ----------------------------------------------------------------------------- #
# create_features() – error cases                                               #
# ----------------------------------------------------------------------------- #

def test_create_features_error(error_case_1, error_case_2, error_case_3, error_case_4):
    """ Tests to ensure create_features function validates inputs correctly and raises appropriate errors."""
    # Test 1 - input not a dataframe
    with pytest.raises(TypeError, match="`tweets` must be a DataFrame."):
        create_features(error_case_1)

    # Test 2 - input has wrong column name for Tweet Text
    with pytest.raises(ValueError, match="The dataframe does not have the right columns."):
        create_features(error_case_2,)

    # Test 3 - input has wrong data type in Date & time column
    with pytest.raises(TypeError, match="`Date & Time` column must be datetime type."):
        create_features(error_case_3)

    # Test 4 - input has wrong data type in Tweet Text column
    with pytest.raises(TypeError, match="`Tweet Text` column must be object type."):
        create_features(error_case_4)

