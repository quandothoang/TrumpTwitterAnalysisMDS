# author: Group 14 - Mailys Guedon, Quan Hoang, Joel Peterson, Li Pu
# date: 2025-12-08

"""
Tests for the season() function in data_utils module.

The season() function maps month numbers (1-12) to season names.
This is a critical function used in the feature engineering pipeline
to create the 'season' column for temporal analysis of tweet frequency.

Run tests with: pytest tests/test_data_utils.py -v
"""

import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import season


class TestSeason:
    """Test suite for the season() function."""

    def test_season_returns_winter_for_january(self):
        """Test that January (month 1) correctly returns 'winter'.

        January is in the middle of winter,
        so this test should expect a result of "winter"
        """
        assert season(1) == 'winter'

    def test_season_returns_spring_for_april(self):
        """Test that April (month 4) correctly returns 'spring'.

        April is the first month of spring,
        so this test should expect a result of "spring"
        """
        assert season(4) == 'spring'

    def test_season_returns_summer_for_july(self):
        """Test that July (month 7) correctly returns 'summer'.

        July is the first month of summer,
        so this test should return "spring".
        """
        assert season(7) == 'summer'

    def test_season_returns_autumn_for_october(self):
        """Test that October (month 10) correctly returns 'autumn'.

        October is the first month of autumn,
        so this test should return "autumn".
        """
        assert season(10) == 'autumn'

    def test_season_boundary_march_is_winter(self):
        """Test that March (month 3) returns 'winter', not 'spring'.

        March is the last month of winter before spring begins in April,
        so this test should return "winter", not "spring".

        """
        assert season(3) == 'winter'

    def test_season_boundary_june_is_spring(self):
        """Test that June (month 6) returns 'spring', not 'summer'.

        June is the last month of spring before summer begins in July,
        so this test should return "spring", not "summer".
        """
        assert season(6) == 'spring'
