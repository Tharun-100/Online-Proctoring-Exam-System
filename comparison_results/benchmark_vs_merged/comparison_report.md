# Benchmark vs Merged Dataset Comparison

## Protocol

- Identical XGBoost hyperparameters for both datasets.
- Stratified 56% training, 14% validation, and 30% test split.
- Exact duplicate rows removed before splitting.
- Imputation, scaling, and categorical encoding fitted only on each training split.
- Decision threshold: 0.5.

## Dataset audit
- Benchmark: 5500 rows in file; 5424 used.
- Merged: 6964 rows in file; 6887 used.
- The current merged file does not contain the previously stated 8,000 rows.

## Test metrics

| dataset   |   rows_in_file |   rows_used |   duplicates_removed |   processed_feature_count |   accuracy |   balanced_accuracy |   precision |   recall |   specificity |     f1 |   roc_auc |   pr_auc |    mcc |

|:----------|---------------:|------------:|---------------------:|--------------------------:|-----------:|--------------------:|------------:|---------:|--------------:|-------:|----------:|---------:|-------:|

| benchmark |           5500 |        5424 |                   76 |                        46 |     0.9945 |              0.9942 |      0.9987 |   0.9895 |        0.9988 | 0.9941 |    0.9995 |   0.9995 | 0.9889 |
| merged    |           6964 |        6887 |                   77 |                        46 |     0.9937 |              0.9936 |      0.9950 |   0.9920 |        0.9953 | 0.9935 |    0.9996 |   0.9996 | 0.9874 |

## Interpretation constraint

These are internal random-split results. They compare in-distribution tabular performance but do not, by themselves, establish generalization to independently recorded real exam videos.