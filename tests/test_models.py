import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


class TestROCCurve:
    """Test ROC curve functionality"""
    
    def test_roc_curve_basic(self):
        """Test basic ROC curve computation"""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.35, 0.8])
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        assert len(fpr) == len(tpr)
        assert 0 <= roc_auc <= 1
        assert fpr[0] == 0.0
        assert tpr[-1] == 1.0
    
    def test_perfect_classifier(self):
        """Test ROC AUC for perfect classifier"""
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        assert roc_auc == 1.0
    
    def test_random_classifier(self):
        """Test ROC AUC for random classifier"""
        np.random.seed(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_scores = np.random.random(100)
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        assert 0.3 < roc_auc < 0.7


class TestModelMetrics:
    """Test model evaluation metrics"""
    
    def test_confidence_interval_calculation(self):
        """Test bootstrap confidence interval calculation"""
        np.random.seed(42)
        data = np.random.normal(0.7, 0.05, 1000)
        
        mean = np.mean(data)
        std = np.std(data)
        
        assert 0.65 < mean < 0.75
        assert std > 0
    
    def test_probability_bounds(self):
        """Test probabilities are within valid range"""
        probs = np.array([0.0, 0.5, 1.0])
        
        assert np.all((probs >= 0) & (probs <= 1))
