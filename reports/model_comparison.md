# Model Comparison

Decision threshold: `0.3` | Test set size: 1,272,416

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| random_forest | 0.9921 | 0.9951 | 0.9936 | 0.9989 |
| xgboost | 0.9838 | 0.9976 | 0.9906 | 0.9984 |

**Selected model: `random_forest`** (highest F1 on the fraud class)
