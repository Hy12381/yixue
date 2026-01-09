# Machine Learning Pipeline - Python Implementation

This is a Python conversion of the R machine learning code for clinical prediction modeling.

## Features

- **Multiple ML Models**: Logistic Regression, SVM, GBM, Neural Network, XGBoost, AdaBoost, LightGBM, CatBoost
- **Comprehensive Evaluation**: ROC curves, calibration curves, confusion matrices
- **Feature Importance**: Variable importance plots for all models
- **SHAP Analysis**: Model interpretation using SHAP values
- **External Validation**: Support for external validation datasets

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from machine_learning import MLPipeline

# Initialize pipeline
pipeline = MLPipeline(
    data_path="data.csv",
    output_dir="./"
)

# Run complete pipeline
pipeline.run_full_pipeline()
```

### With External Validation

```python
pipeline.run_full_pipeline(external_validation_path="外部验证集.csv")
```

## Step-by-Step Usage

If you want more control over the pipeline:

```python
from machine_learning import MLPipeline

# Initialize
pipeline = MLPipeline(data_path="data.csv", output_dir="./results/")

# Load and preprocess data
pipeline.load_and_preprocess_data()

# Split data (70% train, 30% test)
pipeline.split_data(test_size=0.3, external_validation_path="外部验证集.csv")

# Create baseline table
pipeline.create_baseline_table()

# Define models
pipeline.define_models()

# Train models
pipeline.train_models()
pipeline.train_lightgbm()
pipeline.train_catboost()

# Evaluate
pipeline.evaluate_models()

# Generate plots
pipeline.plot_roc_curves()
pipeline.plot_calibration_curves()
pipeline.plot_confusion_matrices()
pipeline.plot_feature_importance()

# SHAP analysis (specify best model)
pipeline.shap_analysis(best_model_name='GBM', n_samples=500)

# Save models
pipeline.save_models()
```

## Data Format

Your CSV file should have:
- **Target variable**: `Result` (0 or 1)
- **Categorical variables**: Binary features (0/1) like `congestive_heart_failure`, `diabetes`, etc.
- **Continuous variables**: Numeric features like `temperature_idx1`, `ph_idx1`, etc.

Example structure:
```
Result,congestive_heart_failure,diabetes,temperature_idx1,ph_idx1,...
0,1,0,36.5,7.4,...
1,0,1,38.2,7.2,...
```

## Output Files

The pipeline generates:

### CSV Files
- `dev.csv` - Training set
- `vad.csv` - Test set
- `训练集Tableone.csv` - Baseline characteristics
- `Train_Evaluation_metrics.csv` - Training metrics
- `Test_Evaluation_metrics.csv` - Test metrics
- `External_Evaluation_metrics.csv` - External validation metrics (if provided)
- `{Model}_important.csv` - Feature importance for each model

### PDF Plots
- `Train_ROC.pdf`, `Test_ROC.pdf`, `External_ROC.pdf` - ROC curves
- `Train_Calibration.pdf`, `Test_Calibration.pdf` - Calibration curves
- `{Dataset}_{Model}_cm_plot.pdf` - Confusion matrices
- `{Model}_important.pdf` - Feature importance plots
- `SHAP_{Model}_*.pdf` - SHAP analysis plots

### Model Files
- `{Model}_model.joblib` - Saved sklearn models
- `LightGBM_model.txt` - LightGBM model
- `CatBoost_model.cbm` - CatBoost model

## Customization

### Modify Hyperparameter Grids

Edit the `define_models()` method in the `MLPipeline` class:

```python
self.model_configs = {
    'Logistic': {
        'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'params': {
            'C': [0.01, 0.1, 1, 10],  # Add more values
            'penalty': ['l1', 'l2']    # Try different penalties
        }
    },
    # ... other models
}
```

### Change Cross-Validation Strategy

In the `train_models()` method:

```python
cv = RepeatedStratifiedKFold(
    n_splits=5,   # Change number of folds
    n_repeats=3,  # Change number of repeats
    random_state=RANDOM_STATE
)
```

### Select Different Best Model for SHAP

```python
pipeline.shap_analysis(best_model_name='XGBoost', n_samples=500)
```

## Key Differences from R Code

1. **Data Handling**: Uses pandas instead of R data frames
2. **Model Training**: Uses scikit-learn's GridSearchCV instead of caret
3. **Visualization**: Uses matplotlib/seaborn instead of ggplot2
4. **SHAP**: Uses Python SHAP library (similar functionality to R's shapper)

## Troubleshooting

### Memory Issues
If you encounter memory errors with SHAP:
```python
pipeline.shap_analysis(best_model_name='GBM', n_samples=100)  # Reduce samples
```

### Encoding Issues
If CSV reading fails:
```python
self.data = pd.read_csv(self.data_path, encoding='utf-8')  # Try different encoding
```

### Missing Values
The pipeline assumes no missing values. If you have missing data, add preprocessing:
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
```

## Notes

- Random seed is set to 52 for reproducibility (matching R code)
- All models use probability predictions for evaluation
- Optimal thresholds are determined using Youden's index
- Feature importance plots are generated for all compatible models

## Support

For issues or questions, please check:
1. Data format matches expected structure
2. All required packages are installed
3. File paths are correct
4. Sufficient memory for SHAP analysis
