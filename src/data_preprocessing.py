"""
Data Preprocessing Module for Fraud Detection System
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

class DataPreprocessor:
    """Class to handle data preprocessing and feature engineering"""

    def __init__(self, categorical_columns=None, auto_detect_categoricals=True):
        self.one_hot_encoder = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.categorical_columns = list(categorical_columns) if categorical_columns else None
        self.auto_detect_categoricals = auto_detect_categoricals
        self.is_fitted = False

    def _resolve_categorical_columns(self, X):
        cat_cols = []
        if self.categorical_columns:
            cat_cols = [col for col in self.categorical_columns if col in X.columns]
        if self.auto_detect_categoricals:
            detected = X.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in detected:
                if col not in cat_cols:
                    cat_cols.append(col)
        self.categorical_columns = cat_cols
        return cat_cols
        
    def fit_transform(self, df, target_column='label'):

        """
        Fit preprocessor and transform data
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """

        df = df.copy()

        # Separate features and target
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])

        # Keep track of numeric columns (exclude categoricals)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Handle categorical columns with one-hot encoding
        cat_cols = self._resolve_categorical_columns(X)
        if cat_cols:
            X[cat_cols] = X[cat_cols].fillna('Unknown').astype(str)
            self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            cat_array = self.one_hot_encoder.fit_transform(X[cat_cols])
            cat_feature_names = self.one_hot_encoder.get_feature_names_out(cat_cols)
            X_cat = pd.DataFrame(cat_array, columns=cat_feature_names, index=X.index)
            X = X.drop(columns=cat_cols)
            X = pd.concat([X, X_cat], axis=1)

        # Handle missing values in numerical columns
        X_numeric = X[numeric_cols] if numeric_cols else X.select_dtypes(include=[np.number])
        X_numeric_imputed = self.imputer.fit_transform(X_numeric)
        X[X_numeric.columns] = X_numeric_imputed
        
        # Scale numerical features (after imputation)
        X_scaled = self.scaler.fit_transform(X_numeric_imputed)
        X[X_numeric.columns] = X_scaled
        
        self.is_fitted = True
        self.feature_columns = list(X.columns)
        return X, y
    
    def transform(self, df, target_column=None):
        """
        Transform new data using fitted preprocessor

        Args:
            df: DataFrame to transform
            target_column: Optional target column to separate

        Returns:
            Transformed features (and target if target_column is provided)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df = df.copy()
        
        # Separate target if provided
        y = None
        if target_column and target_column in df.columns:
            y = df[target_column].copy()
            X = df.drop(columns=[target_column])
        else:
            X = df
        
        # Keep track of numeric columns (exclude categoricals)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Handle categorical columns with one-hot encoding
        cat_cols = [col for col in (self.categorical_columns or []) if col in X.columns]
        if cat_cols and self.one_hot_encoder is not None:
            X[cat_cols] = X[cat_cols].fillna('Unknown').astype(str)
            cat_array = self.one_hot_encoder.transform(X[cat_cols])
            cat_feature_names = self.one_hot_encoder.get_feature_names_out(cat_cols)
            X_cat = pd.DataFrame(cat_array, columns=cat_feature_names, index=X.index)
            X = X.drop(columns=cat_cols)
            X = pd.concat([X, X_cat], axis=1)
        
        # Handle missing values in numerical columns
        X_numeric = X[numeric_cols] if numeric_cols else X.select_dtypes(include=[np.number])
        X_numeric_imputed = self.imputer.transform(X_numeric)
        X[X_numeric.columns] = X_numeric_imputed
        
        # Scale numerical features (after imputation)
        X_scaled = self.scaler.transform(X_numeric_imputed)
        X[X_numeric.columns] = X_scaled
        
        # Ensure same column order as training
        if hasattr(self, 'feature_columns'):
            X = X.reindex(columns=self.feature_columns, fill_value=0)
        
        if y is not None:
            return X, y
        return X
    
    def save(self, filepath):
        """Save preprocessor to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'one_hot_encoder': self.one_hot_encoder,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'categorical_columns': self.categorical_columns,
            'auto_detect_categoricals': self.auto_detect_categoricals,
            'feature_columns': getattr(self, 'feature_columns', None),
            'is_fitted': self.is_fitted
        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load preprocessor from disk"""
        data = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.one_hot_encoder = data.get('one_hot_encoder', None)
        preprocessor.scaler = data['scaler']
        preprocessor.imputer = data['imputer']
        preprocessor.categorical_columns = data['categorical_columns']
        preprocessor.auto_detect_categoricals = data.get('auto_detect_categoricals', True)
        preprocessor.feature_columns = data.get('feature_columns', None)
        preprocessor.is_fitted = data['is_fitted']
        return preprocessor


def load_and_preprocess_data(
    data_path,
    preprocessor=None,
    target_column='label',
    fit_preprocessor=True,
    categorical_columns=None,
    auto_detect_categoricals=True
):
    """
    Load and preprocess data
    
    Args:
        data_path: Path to CSV file
        preprocessor: Optional preprocessor instance
        target_column: Name of target column
        fit_preprocessor: Whether to fit preprocessor (True for training, False for inference)

    Returns:
        Tuple of (X, y, preprocessor)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Initialize preprocessor if not provided
    if preprocessor is None:
        preprocessor = DataPreprocessor(
            categorical_columns=categorical_columns,
            auto_detect_categoricals=auto_detect_categoricals
        )

    # Preprocess data
    if fit_preprocessor:
        X, y = preprocessor.fit_transform(df, target_column=target_column)
    else:
        X, y = preprocessor.transform(df, target_column=target_column)
    
    return X, y, preprocessor