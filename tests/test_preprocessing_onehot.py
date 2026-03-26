import pandas as pd

from src.data_preprocessing import DataPreprocessor


def test_onehot_consistent_columns():
    train_df = pd.DataFrame({
        'head_pose': ['forward', 'left', 'right'],
        'gaze_direction': ['center', 'left', 'right'],
        'face_present': [1, 1, 0],
        'face_conf': [88.5, 76.2, 0.0],
        'label': [0, 1, 0]
    })

    preprocessor = DataPreprocessor(
        categorical_columns=['head_pose', 'gaze_direction'],
        auto_detect_categoricals=True
    )
    
    X_train, _ = preprocessor.fit_transform(train_df, target_column='label')

    assert any(col.startswith('head_pose_') for col in X_train.columns)
    assert any(col.startswith('gaze_direction_') for col in X_train.columns)

    test_df = pd.DataFrame({
        'head_pose': ['up'],  # unseen category
        'gaze_direction': ['center'],
        'face_present': [1],
        'face_conf': [91.0]
    })

    X_test = preprocessor.transform(test_df)

    assert list(X_test.columns) == list(X_train.columns)
    assert X_test.shape[1] == X_train.shape[1]
