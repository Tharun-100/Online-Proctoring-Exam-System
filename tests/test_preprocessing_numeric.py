import numpy as np
import pandas as pd

from src.data_preprocessing import DataPreprocessor


def test_numeric_impute_then_scale_fit_transform():
    df = pd.DataFrame({
        'a': [1.0, np.nan, 3.0],
        'b': [10.0, 10.0, np.nan],
        'label': [0, 1, 0]
    })

    preprocessor = DataPreprocessor(categorical_columns=[], auto_detect_categoricals=False)
    X, _ = preprocessor.fit_transform(df, target_column='label')

    # No NaNs after preprocessing
    assert not np.isnan(X[['a', 'b']].to_numpy()).any()

    # Expected scaling based on imputed values using fitted scaler stats
    expected = np.column_stack([
        np.array([1.0, 2.0, 3.0]),
        np.array([10.0, 10.0, 10.0])
    ])
    expected_scaled = (expected - preprocessor.scaler.mean_) / preprocessor.scaler.scale_

    np.testing.assert_allclose(X[['a', 'b']].to_numpy(), expected_scaled, rtol=1e-6, atol=1e-6)


def test_numeric_impute_then_scale_transform():
    train_df = pd.DataFrame({
        'a': [1.0, 2.0, 3.0],
        'b': [10.0, 11.0, 12.0],
        'label': [0, 1, 0]
    })

    preprocessor = DataPreprocessor(categorical_columns=[], auto_detect_categoricals=False)
    preprocessor.fit_transform(train_df, target_column='label')

    test_df = pd.DataFrame({
        'a': [np.nan, 4.0],
        'b': [13.0, np.nan]
    })

    X_test = preprocessor.transform(test_df)

    assert not np.isnan(X_test[['a', 'b']].to_numpy()).any()

    # Validate transform uses imputed values then scaling
    imputed = preprocessor.imputer.transform(test_df[['a', 'b']])
    expected_scaled = preprocessor.scaler.transform(imputed)
    np.testing.assert_allclose(X_test[['a', 'b']].to_numpy(), expected_scaled, rtol=1e-6, atol=1e-6)
