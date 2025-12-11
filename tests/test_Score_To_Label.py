import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_utils import score_to_label


# ---------------------------
# Common Test Cases
# ---------------------------

class TestScoreToLabelCommon:
    """Tests for normal / typical use cases of score_to_label."""

    def test_positive_sentiment_default_threshold(self):
        """Scores > 0.05 → 'positive'"""
        result = score_to_label(0.5)
        assert result == "positive"

    def test_negative_sentiment_default_threshold(self):
        """Scores < -0.05 → 'negative'"""
        result = score_to_label(-0.3)
        assert result == "negative"

    def test_neutral_sentiment_within_thresholds(self):
        """-0.05 < score < 0.05 → 'neutral'"""
        result = score_to_label(0.02)
        assert result == "neutral"


# ---------------------------
# Custom Threshold Test Cases
# ---------------------------

class TestScoreToLabelCustomThresholds:
    """Tests for custom pos / neg threshold arguments."""

    def test_custom_threshold_neutral(self):
        """
        With pos_threshold=0.1, neg_threshold=-0.1,
        0.08 falls between thresholds → 'neutral'
        """
        result = score_to_label(0.08, pos_threshold=0.1, neg_threshold=-0.1)
        assert result == "neutral"

    def test_custom_threshold_positive(self):
        """
        With pos_threshold=0.1, neg_threshold=-0.1,
        0.15 >= pos_threshold → 'positive'
        """
        result = score_to_label(0.15, pos_threshold=0.1, neg_threshold=-0.1)
        assert result == "positive"


# ---------------------------
# Boundary Value Test Cases
# ---------------------------

class TestScoreToLabelBoundaries:
    """Boundary tests for default thresholds."""

    @pytest.mark.parametrize(
        "score, expected",
        [
            (0.05, "positive"),   # exactly at pos_threshold
            (-0.05, "negative"),  # exactly at neg_threshold
        ],
    )
    def test_boundary_scores(self, score, expected):
        assert score_to_label(score) == expected

class TestScoreToLabelExceptions:
    """Tests for invalid inputs and thresholds."""

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError):
            score_to_label(2.0)

    def test_score_not_numeric(self):
        with pytest.raises(ValueError):
            score_to_label("abc")

    def test_invalid_threshold_types(self):
        with pytest.raises(ValueError):
            score_to_label(0.1, pos_threshold="x")

    def test_invalid_threshold_order(self):
        # neg_threshold >= pos_threshold should raise error
        with pytest.raises(ValueError):
            score_to_label(0.1, pos_threshold=0.05, neg_threshold=0.2)