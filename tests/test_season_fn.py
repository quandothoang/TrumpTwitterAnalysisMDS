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

Run tests with: pytest tests/test_season_fn.py -v
"""

import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_utils import season


class TestSeason:
    """Test expected/common use cases for the season() function."""

    def test_season_returns_winter_for_january(self):
        """Test that January (month 1) correctly returns 'winter'."""
        assert season(1) == 'winter'

    def test_season_returns_spring_for_may(self):
        """Test that May (month 5) correctly returns 'spring'."""
        assert season(5) == 'spring'

    def test_season_returns_summer_for_august(self):
        """Test that August (month 8) correctly returns 'summer'."""
        assert season(8) == 'summer'

    def test_season_returns_autumn_for_november(self):
        """Test that November (month 11) correctly returns 'autumn'."""
        assert season(11) == 'autumn'


class TestSeasonEdgeCases:
    """Test edge (boundary) cases for the season() function."""

    def test_season_boundary_march_is_winter(self):
        """Test that March (month 3), the last winter month, returns 'winter'."""
        assert season(3) == 'winter'

    def test_season_boundary_april_is_spring(self):
        """Test that April (month 4), the first spring month, returns 'spring'."""
        assert season(4) == 'spring'

    def test_season_boundary_june_is_spring(self):
        """Test that June (month 6), the last spring month, returns 'spring'."""
        assert season(6) == 'spring'

    def test_season_boundary_july_is_summer(self):
        """Test that July (month 7), the first summer month, returns 'summer'."""
        assert season(7) == 'summer'

    def test_season_boundary_september_is_summer(self):
        """Test that September (month 9), the last summer month, returns 'summer'."""
        assert season(9) == 'summer'

    def test_season_boundary_october_is_autumn(self):
        """Test that October (month 10), the first autumn month, returns 'autumn'."""
        assert season(10) == 'autumn'

    def test_season_boundary_december_is_autumn(self):
        """Test that December (month 12), the last autumn month, returns 'autumn'."""
        assert season(12) == 'autumn'


class TestSeasonErroneousCases:
    """Test erroneous/adversarial inputs for the season() function."""

    def test_season_raises_error_for_zero(self):
        """Test that month 0 raises a ValueError with message."""
        with pytest.raises(ValueError, match="month must be between 1 and 12"):
            season(0)

    def test_season_raises_error_for_thirteen(self):
        """Test that month 13 raises a ValueError with message."""
        with pytest.raises(ValueError, match="month must be between 1 and 12"):
            season(13)

    def test_season_raises_error_for_negative(self):
        """Test that negative month raises a ValueError with message."""
        with pytest.raises(ValueError, match="month must be between 1 and 12"):
            season(-1)

    def test_season_raises_error_for_string(self):
        """Test that string input raises a TypeError with message."""
        with pytest.raises(TypeError, match="month must be an integer"):
            season("January")

    def test_season_raises_error_for_float(self):
        """Test that float input raises a TypeError with message."""
        with pytest.raises(TypeError, match="month must be an integer"):
            season(1.5)

    def test_season_raises_error_for_none(self):
        """Test that None input raises a TypeError with message."""
        with pytest.raises(TypeError, match="month must be an integer"):
            season(None)
