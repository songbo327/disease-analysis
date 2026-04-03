import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.shap_analysis import analyze_shap_values


class TestSHAPAnalysis:
    """Test SHAP analysis functionality"""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing"""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'age': np.random.randint(40, 70, n),
            'gender': np.random.randint(1, 3, n),
            'height': np.random.randint(150, 190, n),
            'weight': np.random.randint(50, 100, n),
            'ap_hi': np.random.randint(100, 160, n),
            'ap_lo': np.random.randint(60, 100, n),
            'cholesterol': np.random.randint(1, 4, n),
            'gluc': np.random.randint(1, 4, n),
            'smoke': np.random.randint(0, 2, n),
            'alco': np.random.randint(0, 2, n),
            'active': np.random.randint(0, 2, n),
            'bmi': np.random.uniform(20, 35, n),
            'cardio': np.random.randint(0, 2, n)
        })

    def test_shap_analysis_returns_values(self, sample_df):
        """Test SHAP analysis returns expected values"""
        shap_values, X_sample, feature_names = analyze_shap_values(sample_df)

        assert shap_values is not None
        assert X_sample is not None
        assert feature_names is not None

    def test_shap_values_shape(self, sample_df):
        """Test SHAP values have correct shape"""
        shap_values, X_sample, _ = analyze_shap_values(sample_df)

        assert shap_values.shape[0] == X_sample.shape[0]
        assert shap_values.shape[1] == X_sample.shape[1]

    def test_shap_values_numeric(self, sample_df):
        """Test SHAP values are numeric"""
        shap_values, _, _ = analyze_shap_values(sample_df)

        assert np.issubdtype(shap_values.dtype, np.floating)

    def test_shap_importance_csv_saved(self, sample_df):
        """Test SHAP importance CSV is saved"""
        analyze_shap_values(sample_df)

        assert os.path.exists('results/shap_importance.csv')

        shap_df = pd.read_csv('results/shap_importance.csv')
        assert 'Feature' in shap_df.columns
        assert 'Mean_SHAP' in shap_df.columns

    def test_shap_plots_saved(self, sample_df):
        """Test SHAP plots are generated"""
        analyze_shap_values(sample_df)

        assert os.path.exists('plots/shap_summary.png')
        assert os.path.exists('plots/shap_bar.png')