import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.preprocessing import preprocess_data


class TestDataLoader:
    """Test data loading functionality"""
    
    def test_load_valid_csv(self, tmp_path):
        """Test loading valid CSV file"""
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a;b;c\n1;2;3\n4;5;6")
        
        df = load_data(str(csv_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ['a', 'b', 'c']
    
    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error"""
        with pytest.raises(FileNotFoundError):
            load_data("nonexistent.csv")
    
    def test_load_invalid_format(self, tmp_path):
        """Test loading non-CSV file raises error"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("some text")
        
        with pytest.raises(ValueError, match="CSV format"):
            load_data(str(txt_file))
    
    def test_load_empty_csv(self, tmp_path):
        """Test loading empty CSV raises error"""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("a;b;c")
        
        with pytest.raises(ValueError, match="empty"):
            load_data(str(csv_file))


class TestPreprocessing:
    """Test data preprocessing functionality"""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing"""
        return pd.DataFrame({
            'id': [1, 2, 3, 4],
            'age': [21915, 25567, 19320, 22345],
            'gender': [1, 2, 1, 2],
            'height': [168, 175, 160, 170],
            'weight': [70, 80, 60, 75],
            'ap_hi': [120, 130, 110, 140],
            'ap_lo': [80, 85, 70, 90],
            'cholesterol': [1, 2, 1, 3],
            'gluc': [1, 1, 2, 1],
            'smoke': [0, 1, 0, 1],
            'alco': [0, 0, 1, 1],
            'active': [1, 1, 0, 1],
            'cardio': [0, 1, 0, 1]
        })
    
    def test_preprocess_valid_data(self, sample_df):
        """Test preprocessing valid data"""
        result = preprocess_data(sample_df)
        
        assert 'id' not in result.columns
        assert 'bmi' in result.columns
        assert 'age' in result.columns
        assert result['age'].max() < 100
        assert set(result['cardio'].unique()).issubset({0, 1})
    
    def test_preprocess_empty_dataframe(self):
        """Test preprocessing empty dataframe raises error"""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            preprocess_data(empty_df)
    
    def test_preprocess_missing_columns(self):
        """Test preprocessing dataframe with missing columns"""
        incomplete_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        with pytest.raises(ValueError, match="Missing required columns"):
            preprocess_data(incomplete_df)
    
    def test_preprocess_invalid_target(self, sample_df):
        """Test preprocessing with invalid target variable"""
        sample_df['cardio'] = [0, 1, 2, 3]
        with pytest.raises(ValueError, match="must contain only 0 and 1"):
            preprocess_data(sample_df)
    
    def test_bmi_calculation(self, sample_df):
        """Test BMI calculation is correct"""
        result = preprocess_data(sample_df)
        
        expected_bmi = 70 / ((168 / 100) ** 2)
        assert abs(result.iloc[0]['bmi'] - expected_bmi) < 0.01
    
    def test_age_conversion(self, sample_df):
        """Test age conversion from days to years"""
        result = preprocess_data(sample_df)
        
        expected_age = 21915 / 365.25
        assert abs(result.iloc[0]['age'] - expected_age) < 0.01
    
    def test_outlier_removal(self, sample_df):
        """Test outlier removal for blood pressure"""
        sample_df.loc[0, 'ap_hi'] = 300
        sample_df.loc[1, 'ap_lo'] = 20
        
        result = preprocess_data(sample_df)
        
        assert result['ap_hi'].max() <= 250
        assert result['ap_lo'].min() >= 30
    
    def test_duplicate_removal(self, sample_df):
        """Test duplicate removal"""
        duplicate_df = pd.concat([sample_df, sample_df], ignore_index=True)
        result = preprocess_data(duplicate_df)
        
        assert len(result) == len(sample_df)
    
    def test_no_side_effects(self, sample_df):
        """Test preprocessing doesn't modify original dataframe"""
        original_shape = sample_df.shape
        preprocess_data(sample_df)
        
        assert sample_df.shape == original_shape
        assert 'id' in sample_df.columns
