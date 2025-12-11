
import os
import pytest
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import re

from src.data_utils import create_features

# Test case 1 : 3 normal tweets and times

test_case_1_input = pd.DataFrame({
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14),pd.Timestamp(2012,7,14,15,26,17), pd.Timestamp(2011,3,20,1,3,5)],
    'Tweet Text' : ['I am a concerned citizen!', 
                    'There is nothing to worry about, we have heard your complaints and decided to remove them! Be afraid...',
                    'Il fait toujours beau au dessus des nuages. Jécouterai sous la pluie la symphonie des éclairs! Ses cris et ses larmes qui lui faisait tant///...']
})
#print(test_case_1_input)
test_case_1_output = pd.concat([test_case_1_input.copy().reset_index(), pd.DataFrame({
    'length' : [len(test_case_1_input.loc[0,'Tweet Text']), 
                len(test_case_1_input.loc[1,'Tweet Text']),
                len(test_case_1_input.loc[2,'Tweet Text'])],
    'hour'  :  [int(test_case_1_input.loc[0,'Date & Time'].hour), 
                int(test_case_1_input.loc[1,'Date & Time'].hour),
                int(test_case_1_input.loc[2,'Date & Time'].hour)],
    'weekday' : [pd.Timestamp(2012,12,12,12,22,14).weekday(),pd.Timestamp(2012,7,14,15,26,17).weekday(), pd.Timestamp(2011,3,20,1,3,5).weekday()],
    'year' : [pd.Timestamp(2012,12,12,12,22,14).year,pd.Timestamp(2012,7,14,15,26,17).year, pd.Timestamp(2011,3,20,1,3,5).year],
    'month' : [pd.Timestamp(2012,12,12,12,22,14).month,pd.Timestamp(2012,7,14,15,26,17).month, pd.Timestamp(2011,3,20,1,3,5).month],
    'day' : [pd.Timestamp(2012,12,12,12,22,14).day,pd.Timestamp(2012,7,14,15,26,17).day, pd.Timestamp(2011,3,20,1,3,5).day],
    'season' : ['autumn', 'summer', 'winter'],
    'time_of_day' : ['daytime', 'daytime', 'overnight'],
    'avg_word_length' : [4.2, 4.8, 4.8],
    'word_count' : [5, len('There is nothing to worry about, we have heard your complaints and decided to remove them! Be afraid...'.split()), len('Il fait toujours beau au dessus des nuages. Jécouterai sous la pluie la symphonie des éclairs! Ses cris et ses larmes qui lui faisait tant///...'.split())],
    'punctuation_count' : [1, 5, 8]
})], axis=1)


#print(test_case_1_output)

# Edge case 1 : empty letter tweet

edge_case_1_input = pd.DataFrame({
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14)],
    'Tweet Text' : ['']
})
#print(edge_case_1_input)

edge_case_1_output = pd.DataFrame({
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14)],
    'Tweet Text' : [' '],
    'length':[0], 
    'hour' : [12],
    'weekday' : [2],
    'year' : [2012],
    'month' : [12],
    'day' : [12],
    'season' : ['autumn'],
    'time_of_day' : ['daytime'],
    'avg_word_length' : [0],
    'word_count' : [0],
    'punctuation_count' : [0]
})

# Edge case 2 : only punctuation

edge_case_2_input = pd.DataFrame({
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14)],
    'Tweet Text' : ['...///...^']
})
#print(edge_case_2_input)

edge_case_1_output = pd.DataFrame({
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14)],
    'Tweet Text' : ['I'],
    'length':[1], 
    'hour' : [12],
    'weekday' : [2],
    'year' : [2012],
    'month' : [12],
    'day' : [12],
    'season' : ['autumn'],
    'time_of_day' : ['daytime'],
    'avg_word_length' : [1],
    'word_count' : [1],
    'punctuation_count' : [0]
})

# Error case 2 : input not dataframe

error_case_2_input = {
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14)],
    'Tweet Text' : ['I am a concerned citizen!']
}
#print(error_case_2_input)

# Error case 3 : no Date & time or Tweet Text column
error_case_3_input = pd.DataFrame({
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14)],
    'Tweet' : ['I am a concerned citizen!']
})
#print(error_case_3_input)

# Error case 4 :  Date & time  not timestamp 
error_case_4_input = pd.DataFrame({
    'Date & Time'  : ['2012/12/24'],
    'Tweet Text' : ['I am a concerned citizen!']
})
# Error case 5 :  Date & time  not timestamp 
error_case_5_input = pd.DataFrame({
    'Date & Time'  : [pd.Timestamp(2012,12,12,12,22,14)],
    'Tweet Text' : [1]
})
#print(error_case_4_input)

# Tests :
def create_features_test_normal():
    """ Tests to ensure create_features function correctly adds features."""
    results = create_features(test_case_1_input)
    print(results)
    # Verify return type 
    assert isinstance(results, pd.DataFrame)

    # Verify that the output dataframe is as expected
    pd.testing.assert_frame_equal(test_case_1_output, results, check_dtype = False)



#print(create_features_test_normal())

def create_features_test_edge():
    """ Tests to ensure create_features function works correctly when the string is empty or when there is only punctuation."""

    # Test empty tweet - should return 0 for length, avg_word_length, word_count, punctuation_count
    result_empty = create_features(edge_case_1_input)
    assert result_empty.iloc[0, 11:].all() == 0 
    
    # Test punctuation tweet - should return 0 for word_count and avg_word_length
    result_punct = create_features(edge_case_2_input)
    assert result_punct.iloc[0, 11:13].all() == 0 

create_features_test_edge()


def create_features_test_error():
    """ Tests to ensure create_features function validates inputs correctly and raises appropriate errors."""
    # Test 1 - input not a dataframe
    with pytest.raises(TypeError, match="`tweets` must be a DataFrame."):
        create_features(error_case_2_input)

    # Test 2 - input has wrong column name
    with pytest.raises(ValueError, match=f"The dataframe is missing columns."):
        create_features(error_case_3_input)

    # Test 3 - input has wrong data type in Date & time column
    with pytest.raises(TypeError, match="`Date & Time` column must be datetime type."):
        create_features(error_case_4_input)

    # Test 4 - input has wrong data type in Tweet Text column
    with pytest.raises(TypeError, match="`Tweet Text` column must be object type."):
        create_features(error_case_5_input)

create_features_test_error()