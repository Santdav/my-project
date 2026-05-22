import pytest 
from src.preprocess import load_raw,engineer_features,encode
import pandas as pd


def test_load_raw():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    assert load_raw("data/WA_Fn-UseC_-Telco-Customer-Churn.csv").shape == df.shape

