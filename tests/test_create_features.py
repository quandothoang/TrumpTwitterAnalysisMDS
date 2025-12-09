
import os
import pytest
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# Test case 1

test_case_1_input = pd.DataFrame({
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14),pd.Timestamp(2012,7,14,15,26,17), pd.Timestamp(2011,3,20,1,3,5)],
    'Tweet Text' : ['I am a concerned citizen!', 
                    'There is nothing to worry about, we have heard your complaints and decided to remove them! Be afraid...',
                      'Il fait toujours beau au dessus des nuages. Jécouterai sous la pluie la symphonie des éclairs! Ses cris et ses larmes qui lui faisait tant///...']
})
print(test_case_1_input)
test_case_1_output = pd.DataFrame({
    'Date & Time' : [pd.Timestamp(2012,12,12,12,22,14),pd.Timestamp(2012,7,14,15,26,17), pd.Timestamp(2011,3,20,1,3,5)],
    'Tweet Text' : ['I am a concerned citizen!', 
                    'There is nothing to worry about, we have heard your complaints and decided to remove them! Be afraid...',
                    'Il fait toujours beau au dessus des nuages. Jécouterai sous la pluie la symphonie des éclairs! Ses cris et ses larmes qui lui faisait tant///...'],
    'length' : [len('I am a concerned citizen!'), 
                len('There is nothing to worry about, we have heard your complaints and decided to remove them! Be afraid...'),
                len('Il fait toujours beau au dessus des nuages. Jécouterai sous la pluie la symphonie des éclairs! Ses cris et ses larmes qui lui faisait tant///...')],
    'hour'  :  [pd.Timestamp(2012,12,12,12,22,14).hour,pd.Timestamp(2012,7,14,15,26,17).hour, pd.Timestamp(2011,3,20,1,3,5).hour],
    'weekday' : [pd.Timestamp(2012,12,12,12,22,14).weekday(),pd.Timestamp(2012,7,14,15,26,17).weekday(), pd.Timestamp(2011,3,20,1,3,5).weekday()],
    'year' : [pd.Timestamp(2012,12,12,12,22,14).year,pd.Timestamp(2012,7,14,15,26,17).year, pd.Timestamp(2011,3,20,1,3,5).year],
    'month' : [pd.Timestamp(2012,12,12,12,22,14).month,pd.Timestamp(2012,7,14,15,26,17).month, pd.Timestamp(2011,3,20,1,3,5).month],
    'day' : [pd.Timestamp(2012,12,12,12,22,14).day,pd.Timestamp(2012,7,14,15,26,17).day, pd.Timestamp(2011,3,20,1,3,5).day],
    'season' : ['autumn', 'summer', 'winter'],
    'time_of_day' : ['daytime', 'daytime', 'overnight'],
    'avg_word_length' : [4.2, 4.8, 4.8],
    'word_count' : [5, len('There is nothing to worry about, we have heard your complaints and decided to remove them! Be afraid...'.split()), len('Il fait toujours beau au dessus des nuages. Jécouterai sous la pluie la symphonie des éclairs! Ses cris et ses larmes qui lui faisait tant///...'.split())],
    'punctuation_count' : [1, 5, 8]
})

print(test_case_1_output)

