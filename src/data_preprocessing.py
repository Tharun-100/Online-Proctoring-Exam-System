"""
Data Preprocessing Module for Fraud Detection System
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os


class DataPreprocessor:
    """Class to handle data preprocessing and feature engineering"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.categorical_columns = ['head_pose', 'gaze_direction']
        self.is_fitted = False
        
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
        
        # Handle categorical columns
        for col in self.categorical_columns:
            if col in X.columns:
                # Fill missing values with 'Unknown'
                X[col] = X[col].fillna('Unknown')
                # Encode categorical variables
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                else:
                    # Handle unseen categories during transform (shouldn't happen in fit_transform, but safe guard)
                    X[col] = X[col].astype(str)
                    known_classes = set(self.label_encoders[col].classes_)
                    default_class = self.label_encoders[col].classes_[0]
                    X[col] = X[col].apply(lambda x: x if x in known_classes else default_class)
                    X[col] = self.label_encoders[col].transform(X[col])
        
        # Handle missing values in numerical columns
        X_numeric = X.select_dtypes(include=[np.number])
        X_numeric_imputed = self.imputer.fit_transform(X_numeric)
        X[X_numeric.columns] = X_numeric_imputed
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X.select_dtypes(include=[np.number]))
        X[X.select_dtypes(include=[np.number]).columns] = X_scaled
        
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
        
        # Handle categorical columns
        for col in self.categorical_columns:
            if col in X.columns:
                # Fill missing values with 'Unknown'
                X[col] = X[col].fillna('Unknown')
                # Transform categorical variables
                if col in self.label_encoders:
                    X[col] = X[col].astype(str)
                    known_classes = set(self.label_encoders[col].classes_)
                    # Map unseen categories to the first known class (or most common)
                    default_class = self.label_encoders[col].classes_[0]
                    X[col] = X[col].apply(lambda x: x if x in known_classes else default_class)
                    X[col] = self.label_encoders[col].transform(X[col])
        
        # Handle missing values in numerical columns
        X_numeric = X.select_dtypes(include=[np.number])
        X_numeric_imputed = self.imputer.transform(X_numeric)
        X[X_numeric.columns] = X_numeric_imputed
        
        # Scale numerical features
        X_scaled = self.scaler.transform(X.select_dtypes(include=[np.number]))
        X[X.select_dtypes(include=[np.number]).columns] = X_scaled
        
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
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'categorical_columns': self.categorical_columns,
            'feature_columns': getattr(self, 'feature_columns', None),
            'is_fitted': self.is_fitted
        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load preprocessor from disk"""
        data = joblib.load(filepath)
        preprocessor = cls()
        preprocessor.label_encoders = data['label_encoders']
        preprocessor.scaler = data['scaler']
        preprocessor.imputer = data['imputer']
        preprocessor.categorical_columns = data['categorical_columns']
        preprocessor.feature_columns = data.get('feature_columns', None)
        preprocessor.is_fitted = data['is_fitted']
        return preprocessor


def load_and_preprocess_data(data_path, preprocessor=None, target_column='label', fit_preprocessor=True):
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
        preprocessor = DataPreprocessor()
    
    # Preprocess data
    if fit_preprocessor:
        X, y = preprocessor.fit_transform(df, target_column=target_column)
    else:
        X, y = preprocessor.transform(df, target_column=target_column)
    
    return X, y, preprocessor

