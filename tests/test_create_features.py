# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-08

"""
Tests for the season() function in data_utils module.

The season() function maps month numbers (1-12) to season names.
This is a critical function used in the feature engineering pipeline
to create the 'season' column for temporal analysis of tweet frequency.

Test categories:
1. Expected/common use cases - Normal month inputs (1-12)
2. Edge cases - Boundary months between seasons
3. Erroneous/adversarial use cases - Invalid inputs

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
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14),pd.Timestamp(2012,7,14,15,26,17), pd.Timestamp(2011,3,20,1,3,5)],
    'Tweet Text' : ['I am a concerned citizen!', 
                    'There is nothing to worry about, we have heard your complaints and decided to remove them! Be afraid...',
                    'Il fait toujours beau au dessus des nuages. Jécouterai sous la pluie la symphonie des éclairs! Ses cris et ses larmes qui lui faisait tant///...']
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
    'season' : ['autumn', 
                'summer', 
                'winter'],
    'time_of_day' : ['daytime', 
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

