"""
Machine Learning Pipeline - Converted from R
Multi-model training, evaluation, and interpretation
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix, 
                             classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, auc)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder

# Additional libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import shap

# Set random seed for reproducibility
RANDOM_STATE = 52
np.random.seed(RANDOM_STATE)

WEB_FEATURES = [
    "invasive_ventilator_flag",
    "apsiii_min",
    "temperature_idx1",
    "ph_idx1",
    "noninvasive_ventilator_flag",
    "cerebrovascular_disease",
    "severe_liver_disease",
]

WEB_FEATURE_LABELS_ZH = {
    "invasive_ventilator_flag": "有创呼吸机（invasive_ventilator）",
    "apsiii_min": "APACHE III评分最小值（APACHE III score minimum）",
    "temperature_idx1": "体温（temperature）",
    "ph_idx1": "pH值（pH）",
    "noninvasive_ventilator_flag": "无创呼吸机（noninvasive_ventilator）",
    "cerebrovascular_disease": "脑血管病（cerebrovascular disease）",
    "severe_liver_disease": "严重肝脏疾病（severe liver disease）",
}

def _safe_read_csv(path):
    for encoding in ("gbk", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="gbk")

def _coerce_numeric_series(s):
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().all():
        return numeric
    codes = s.astype("category").cat.codes.replace(-1, np.nan)
    if codes.isna().any():
        mode = codes.mode(dropna=True)
        fill_value = int(mode.iloc[0]) if len(mode) > 0 else 0
        codes = codes.fillna(fill_value)
    return codes.astype(int)

def train_web_delirium_model(
    data_path,
    output_dir,
    external_validation_path=None,
    feature_names=None,
):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if not output_dir.endswith(os.sep):
        output_dir += os.sep

    feature_names = feature_names or WEB_FEATURES
    data = _safe_read_csv(data_path)
    if "Result" not in data.columns:
        raise ValueError("CSV 缺少 Result 列")

    missing = [c for c in feature_names if c not in data.columns]
    if missing:
        raise ValueError(f"CSV 缺少特征列: {missing}")

    X = data[feature_names].copy()
    for c in feature_names:
        X[c] = _coerce_numeric_series(X[c])

    y = _coerce_numeric_series(data["Result"])
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=5000)
    model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    metrics = [
        {"Dataset": "Train", "AUC": float(roc_auc_score(y_train, train_proba))},
        {"Dataset": "Test", "AUC": float(roc_auc_score(y_test, test_proba))},
    ]

    if external_validation_path and os.path.exists(external_validation_path):
        external_data = _safe_read_csv(external_validation_path)
        if "Result" in external_data.columns:
            X_ext = external_data[feature_names].copy()
            for c in feature_names:
                X_ext[c] = _coerce_numeric_series(X_ext[c])
            y_ext = _coerce_numeric_series(external_data["Result"]).astype(int)
            ext_proba = model.predict_proba(X_ext)[:, 1]
            metrics.append({"Dataset": "External", "AUC": float(roc_auc_score(y_ext, ext_proba))})

    model_bundle = {
        "model_type": "logistic_regression",
        "model": model,
        "feature_names": feature_names,
        "labels_zh": {k: WEB_FEATURE_LABELS_ZH.get(k, k) for k in feature_names},
        "trained_at": pd.Timestamp.now().isoformat(),
    }

    model_path = f"{output_dir}delirium_web_model.joblib"
    joblib.dump(model_bundle, model_path)
    pd.DataFrame(metrics).to_csv(f"{output_dir}delirium_web_model_auc.csv", index=False)

    return model_path

def create_web_app(model_path):
    from flask import Flask, request

    model_path = os.path.abspath(model_path)
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    labels_zh = bundle.get("labels_zh", {})

    app = Flask(__name__)

    def render_form(values=None, result=None, error=None):
        values = values or {}
        rows = []
        for name in feature_names:
            label = labels_zh.get(name, name)
            value = values.get(name, "")
            input_type = "number"
            step = "1"
            min_attr = ""
            if name in {
                "invasive_ventilator_flag",
                "noninvasive_ventilator_flag",
                "cerebrovascular_disease",
                "severe_liver_disease",
            }:
                min_attr = ' min="0" max="1"'
                step = "1"
            elif name == "temperature_idx1":
                step = "0.01"
            elif name == "ph_idx1":
                step = "0.01"
            else:
                step = "0.01"

            rows.append(
                f"""
                <div style="margin:10px 0;">
                  <label style="display:block;font-weight:600;">{label}</label>
                  <input name="{name}" type="{input_type}" step="{step}" value="{value}"{min_attr} style="width:320px;padding:8px;">
                </div>
                """
            )

        result_html = ""
        if result is not None:
            result_html = f"""
            <div style="margin-top:16px;padding:12px;border:1px solid #ddd;background:#f7fbff;">
              <div style="font-weight:700;">预测结果</div>
              <div style="margin-top:6px;">ICU期间发生谵妄的概率：<span style="font-size:20px;">{result:.3%}</span></div>
            </div>
            """

        error_html = ""
        if error:
            error_html = f"""
            <div style="margin-top:16px;padding:12px;border:1px solid #f5c2c7;background:#f8d7da;color:#842029;">
              {error}
            </div>
            """

        return f"""
        <html>
          <head>
            <meta charset="utf-8">
            <title>COPD 老年 ICU 谵妄风险预测</title>
          </head>
          <body style="font-family:Arial,Helvetica,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;">
            <h2>COPD 老年患者 ICU 期间谵妄概率预测</h2>
            <div style="color:#555;margin-bottom:12px;">输入变量后点击“预测”。0/1 变量：0=否，1=是。</div>
            <form method="post">
              {''.join(rows)}
              <button type="submit" style="padding:10px 18px;font-size:16px;">预测</button>
            </form>
            {error_html}
            {result_html}
            <div style="margin-top:22px;color:#666;font-size:12px;">模型文件：{model_path}</div>
          </body>
        </html>
        """

    @app.get("/")
    def index_get():
        return render_form()

    @app.post("/")
    def index_post():
        values = {}
        try:
            x = []
            for name in feature_names:
                raw = request.form.get(name, "").strip()
                if raw == "":
                    raise ValueError(f"请填写：{labels_zh.get(name, name)}")
                val = float(raw)
                if name in {
                    "invasive_ventilator_flag",
                    "noninvasive_ventilator_flag",
                    "cerebrovascular_disease",
                    "severe_liver_disease",
                }:
                    if val not in (0.0, 1.0):
                        raise ValueError(f"{labels_zh.get(name, name)} 只能填 0 或 1")
                values[name] = raw
                x.append(val)

            X_input = pd.DataFrame([x], columns=feature_names)
            proba = float(model.predict_proba(X_input)[0, 1])
            return render_form(values=values, result=proba)
        except Exception as e:
            return render_form(values=values, error=str(e))

    return app

class MLPipeline:
    def __init__(self, data_path, output_dir="./"):
        """
        Initialize ML Pipeline
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV data file
        output_dir : str
            Directory for saving results
        """
        self.data_path = data_path
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.output_dir.endswith(os.sep):
            self.output_dir += os.sep
        self.models = {}
        self.results = {}
        self.predictions = {}

    def _load_csv(self, path):
        for encoding in ("gbk", "utf-8-sig", "utf-8"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, encoding="gbk")

    def _align_features(self, df, reference_columns):
        aligned = df.copy()
        for col in reference_columns:
            if col not in aligned.columns:
                aligned[col] = 0
        aligned = aligned[reference_columns]
        for col in aligned.columns:
            if aligned[col].dtype.name in {"category", "object"}:
                numeric = pd.to_numeric(aligned[col], errors="coerce")
                if numeric.notna().all():
                    aligned[col] = numeric
                else:
                    codes = aligned[col].astype("category").cat.codes.replace(-1, np.nan)
                    if codes.isna().any():
                        mode = codes.mode(dropna=True)
                        fill_value = int(mode.iloc[0]) if len(mode) > 0 else 0
                        codes = codes.fillna(fill_value)
                    aligned[col] = codes.astype(int)
        return aligned
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset"""
        print("Loading data...")
        self.data = self._load_csv(self.data_path)
        
        # Define categorical columns (adjust based on your data)
        categorical_cols = [
            'congestive_heart_failure', 'peripheral_vascular_disease',
            'cerebrovascular_disease', 'diabetes', 'severe_liver_disease',
            'rel_disease', 'hypertensive', 'invasive_ventilator_flag',
            'noninvasive_ventilator_flag', 'gender'
        ]
        
        for col in categorical_cols:
            if col in self.data.columns:
                numeric = pd.to_numeric(self.data[col], errors='coerce')
                if numeric.notna().all():
                    self.data[col] = numeric
                else:
                    codes = self.data[col].astype('category').cat.codes.replace(-1, np.nan)
                    if codes.isna().any():
                        mode = codes.mode(dropna=True)
                        fill_value = int(mode.iloc[0]) if len(mode) > 0 else 0
                        codes = codes.fillna(fill_value)
                    self.data[col] = codes.astype(int)
        
        # Target variable
        self.data['Result'] = self.data['Result'].astype('category')
        
        print(f"Data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
    def split_data(self, test_size=0.3, external_validation_path=None):
        """Split data into train, test, and optional external validation sets"""
        print("\nSplitting data...")
        
        X = self.data.drop('Result', axis=1)
        y = self.data['Result']
        X = self._align_features(X, X.columns.tolist())
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=RANDOM_STATE, stratify=y_encoded
        )
        
        # Save train and test sets
        train_df = pd.concat([self.X_train, pd.Series(self.y_train, index=self.X_train.index, name='Result')], axis=1)
        test_df = pd.concat([self.X_test, pd.Series(self.y_test, index=self.X_test.index, name='Result')], axis=1)
        
        train_df.to_csv(f"{self.output_dir}dev.csv", index=False)
        test_df.to_csv(f"{self.output_dir}vad.csv", index=False)
        
        print(f"Train set: {self.X_train.shape}, Test set: {self.X_test.shape}")
        
        # Load external validation if provided
        external_validation_path = os.path.abspath(external_validation_path) if external_validation_path else None
        if external_validation_path and os.path.exists(external_validation_path):
            self.external_data = self._load_csv(external_validation_path)
            self.X_external = self.external_data.drop('Result', axis=1)
            self.X_external = self._align_features(self.X_external, self.X_train.columns.tolist())
            self.y_external = self.label_encoder.transform(self.external_data['Result'])
            print(f"External validation set: {self.X_external.shape}")
        else:
            if external_validation_path and not os.path.exists(external_validation_path):
                print(f"External validation file not found, skipping: {external_validation_path}")
            self.X_external = None
            self.y_external = None
            
    def create_baseline_table(self):
        """Create baseline characteristics table"""
        print("\nCreating baseline table...")
        
        train_data = pd.concat([self.X_train, pd.Series(self.y_train, index=self.X_train.index, name='Result')], axis=1)
        
        # Basic statistics
        summary = train_data.describe(include='all').T
        summary.to_csv(f"{self.output_dir}训练集Tableone.csv")
        
        print("Baseline table saved.")
        
    def define_models(self):
        """Define models with hyperparameter grids"""
        print("\nDefining models...")
        
        self.model_configs = {
            'Logistic': {
                'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2']
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=RANDOM_STATE),
                'params': {
                    'C': [0.1, 0.5],
                    'gamma': [0.1, 0.01, 0.001],
                    'kernel': ['rbf']
                }
            },
            'GBM': {
                'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [100],
                    'max_depth': [2, 3],
                    'learning_rate': [0.1, 0.01],
                    'min_samples_leaf': [5]
                }
            },
            'NeuralNetwork': {
                'model': MLPClassifier(random_state=RANDOM_STATE, max_iter=1000),
                'params': {
                    'hidden_layer_sizes': [(3,), (4,), (5,)],
                    'alpha': [0.6]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'n_estimators': [10],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.1, 0.01, 0.001],
                    'gamma': [0.5],
                    'colsample_bytree': [0.5],
                    'subsample': [0.6]
                }
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [2],
                    'learning_rate': [1.0]
                }
            }
        }
        
    def train_models(self):
        """Train all models with cross-validation"""
        print("\nTraining models...")
        
        # Cross-validation strategy
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=RANDOM_STATE)
        
        self.predictions['Train'] = pd.DataFrame({'Result': self.y_train})
        self.predictions['Test'] = pd.DataFrame({'Result': self.y_test})
        if self.X_external is not None:
            self.predictions['External'] = pd.DataFrame({'Result': self.y_external})
        
        for model_name, config in self.model_configs.items():
            print(f"\nTraining {model_name}...")
            
            # Grid search
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Store best model
            self.models[model_name] = grid_search.best_estimator_
            
            # Predictions
            self.predictions['Train'][model_name] = grid_search.predict_proba(self.X_train)[:, 1]
            self.predictions['Test'][model_name] = grid_search.predict_proba(self.X_test)[:, 1]
            if self.X_external is not None:
                self.predictions['External'][model_name] = grid_search.predict_proba(self.X_external)[:, 1]
            
            print(f"{model_name} - Best params: {grid_search.best_params_}")
            print(f"{model_name} - Best CV AUC: {grid_search.best_score_:.3f}")
            
    def train_lightgbm(self):
        """Train LightGBM model"""
        print("\nTraining LightGBM...")
        
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        test_data = lgb.Dataset(self.X_test, label=self.y_test, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 1.0,
            'num_threads': 2,
            'force_col_wise': True,
            'verbose': -1
        }
        
        self.models['LightGBM'] = lgb.train(
            params,
            train_data,
            num_boost_round=5,
            valid_sets=[test_data],
            callbacks=[lgb.early_stopping(stopping_rounds=3)]
        )
        
        self.predictions['Train']['LightGBM'] = self.models['LightGBM'].predict(self.X_train)
        self.predictions['Test']['LightGBM'] = self.models['LightGBM'].predict(self.X_test)
        if self.X_external is not None:
            self.predictions['External']['LightGBM'] = self.models['LightGBM'].predict(self.X_external)
            
    def train_catboost(self):
        """Train CatBoost model"""
        print("\nTraining CatBoost...")
        
        train_pool = Pool(self.X_train, self.y_train)
        test_pool = Pool(self.X_test, self.y_test)
        
        self.models['CatBoost'] = CatBoostClassifier(
            iterations=100,
            depth=5,
            learning_rate=0.03,
            random_seed=RANDOM_STATE,
            verbose=0
        )
        
        self.models['CatBoost'].fit(train_pool, eval_set=test_pool, use_best_model=True)
        
        self.predictions['Train']['CatBoost'] = self.models['CatBoost'].predict_proba(self.X_train)[:, 1]
        self.predictions['Test']['CatBoost'] = self.models['CatBoost'].predict_proba(self.X_test)[:, 1]
        if self.X_external is not None:
            self.predictions['External']['CatBoost'] = self.models['CatBoost'].predict_proba(self.X_external)[:, 1]

    def evaluate_models(self):
        """Evaluate all models on all datasets"""
        print("\nEvaluating models...")
        
        for dataset_name in self.predictions.keys():
            print(f"\n{'='*50}")
            print(f"Evaluating on {dataset_name} set")
            print(f"{'='*50}")
            
            y_true = self.predictions[dataset_name]['Result']
            results_df = []
            
            for model_name in self.models.keys():
                y_pred_proba = self.predictions[dataset_name][model_name]
                
                # Calculate ROC curve and AUC
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                # Find optimal threshold (Youden's index)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                # Binary predictions
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                sensitivity = recall_score(y_true, y_pred)
                specificity = recall_score(y_true, y_pred, pos_label=0)
                precision = precision_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                
                results_df.append({
                    'Model': model_name,
                    'Threshold': f"{optimal_threshold:.3f}",
                    'AUC': f"{roc_auc:.3f}",
                    'Accuracy': f"{accuracy:.3f}",
                    'Sensitivity': f"{sensitivity:.3f}",
                    'Specificity': f"{specificity:.3f}",
                    'Precision': f"{precision:.3f}",
                    'F1': f"{f1:.3f}"
                })
                
                print(f"\n{model_name}:")
                print(f"  AUC: {roc_auc:.3f}")
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  Sensitivity: {sensitivity:.3f}")
                print(f"  Specificity: {specificity:.3f}")
            
            # Save results
            results_df = pd.DataFrame(results_df)
            results_df.to_csv(f"{self.output_dir}{dataset_name}_Evaluation_metrics.csv", index=False)
            
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        print("\nPlotting ROC curves...")
        
        for dataset_name in self.predictions.keys():
            plt.figure(figsize=(10, 8))
            
            y_true = self.predictions[dataset_name]['Result']
            
            for model_name in self.models.keys():
                y_pred_proba = self.predictions[dataset_name][model_name]
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            plt.title(f'ROC Curve - {dataset_name} Set', fontsize=15, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}{dataset_name}_ROC.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            
    def plot_calibration_curves(self):
        """Plot calibration curves for all models"""
        print("\nPlotting calibration curves...")
        
        for dataset_name in self.predictions.keys():
            plt.figure(figsize=(10, 8))
            
            y_true = self.predictions[dataset_name]['Result']
            
            for model_name in self.models.keys():
                y_pred_proba = self.predictions[dataset_name][model_name]
                
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    y_true, y_pred_proba, n_bins=10, strategy='uniform'
                )
                
                plt.plot(mean_predicted_value, fraction_of_positives, 
                        marker='o', lw=2, label=model_name)
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')
            plt.xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
            plt.ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
            plt.title(f'Calibration Curve - {dataset_name} Set', fontsize=15, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}{dataset_name}_Calibration.pdf", dpi=300, bbox_inches='tight')
            plt.close()
            
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print("\nPlotting confusion matrices...")
        
        for dataset_name in self.predictions.keys():
            y_true = self.predictions[dataset_name]['Result']
            
            for model_name in self.models.keys():
                y_pred_proba = self.predictions[dataset_name][model_name]
                
                # Find optimal threshold
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
                plt.xlabel('Predicted', fontsize=12, fontweight='bold')
                plt.ylabel('Actual', fontsize=12, fontweight='bold')
                plt.title(f'{model_name} - {dataset_name} Set', fontsize=15, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}{dataset_name}_{model_name}_cm_plot.pdf", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
    def plot_feature_importance(self):
        """Plot feature importance for all models"""
        print("\nPlotting feature importance...")
        
        feature_names = self.X_train.columns
        
        for model_name, model in self.models.items():
            try:
                # Get feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_[0])
                elif model_name == 'LightGBM':
                    importance = model.feature_importance(importance_type='gain')
                else:
                    continue
                
                # Create dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=True)
                
                # Save to CSV
                importance_df.to_csv(f"{self.output_dir}{model_name}_important.csv", index=False)
                
                # Plot
                plt.figure(figsize=(10, 8))
                plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
                plt.xlabel('Importance Scores', fontsize=12, fontweight='bold')
                plt.ylabel('Features', fontsize=12, fontweight='bold')
                plt.title(f'{model_name} Feature Importance', fontsize=15, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}{model_name}_important.pdf", dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Could not plot feature importance for {model_name}: {e}")
                
    def shap_analysis(self, best_model_name='GBM', n_samples=500):
        """Perform SHAP analysis on the best model"""
        print(f"\nPerforming SHAP analysis for {best_model_name}...")
        
        if best_model_name not in self.models:
            print(f"Model {best_model_name} not found!")
            return
        
        model = self.models[best_model_name]
        
        # Sample data for SHAP
        X_sample = self.X_train.iloc[:n_samples]
        
        try:
            # Create SHAP explainer
            if best_model_name in ['LightGBM', 'XGBoost', 'GBM', 'CatBoost']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, X_sample)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # For binary classification, get positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Summary plot (beeswarm)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.title(f'{best_model_name} SHAP Summary', fontsize=15, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}SHAP_{best_model_name}_importance_beeswarm.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.title(f'{best_model_name} SHAP Importance', fontsize=15, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}SHAP_{best_model_name}_importance_bar.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Waterfall plot for first sample
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                 base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
                                                 data=X_sample.iloc[0],
                                                 feature_names=X_sample.columns.tolist()),
                               show=False)
            plt.title(f'{best_model_name} SHAP Waterfall', fontsize=15, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}SHAP_{best_model_name}_waterfall.pdf", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("SHAP analysis completed!")
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            
    def save_models(self):
        """Save all trained models"""
        print("\nSaving models...")
        
        for model_name, model in self.models.items():
            if model_name in ['LightGBM', 'CatBoost']:
                # These models have their own save methods
                if model_name == 'LightGBM':
                    model_path = f"{self.output_dir}{model_name}_model.txt"
                    try:
                        model.save_model(model_path)
                    except Exception:
                        model_str = model.model_to_string()
                        with open(model_path, "w", encoding="utf-8") as f:
                            f.write(model_str)
                elif model_name == 'CatBoost':
                    model.save_model(f"{self.output_dir}{model_name}_model.cbm")
            else:
                joblib.dump(model, f"{self.output_dir}{model_name}_model.joblib")
        
        print("Models saved!")
        
    def run_full_pipeline(self, external_validation_path=None):
        """Run the complete ML pipeline"""
        print("="*60)
        print("Starting Machine Learning Pipeline")
        print("="*60)
        
        # Load and preprocess
        self.load_and_preprocess_data()
        
        # Split data
        self.split_data(external_validation_path=external_validation_path)
        
        # Baseline table
        self.create_baseline_table()
        
        # Define and train models
        self.define_models()
        self.train_models()
        self.train_lightgbm()
        self.train_catboost()
        
        # Evaluate
        self.evaluate_models()
        
        # Visualizations
        self.plot_roc_curves()
        self.plot_calibration_curves()
        self.plot_confusion_matrices()
        self.plot_feature_importance()
        
        # SHAP analysis
        self.shap_analysis(best_model_name='GBM')
        
        # Save models
        self.save_models()
        
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)


if __name__ == "__main__":
    default_data_path = r"e:\桌面图标\新建文件夹\相关图和表\data.csv"
    default_external_path = r"e:\桌面图标\新建文件夹\外部验证\外部验证集.csv"
    default_output_dir = r"e:\桌面图标\新建文件夹\新建文件夹"

    mode = sys.argv[1].lower() if len(sys.argv) >= 2 else "pipeline"

    if mode == "train_web":
        data_path = sys.argv[2] if len(sys.argv) >= 3 else default_data_path
        external_path = sys.argv[3] if len(sys.argv) >= 4 else default_external_path
        output_dir = sys.argv[4] if len(sys.argv) >= 5 else default_output_dir
        model_path = train_web_delirium_model(
            data_path=data_path,
            output_dir=output_dir,
            external_validation_path=external_path,
        )
        print(f"Web 模型已保存: {model_path}")
    elif mode == "serve":
        model_path = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(default_output_dir, "delirium_web_model.joblib")
        host = sys.argv[3] if len(sys.argv) >= 4 else "127.0.0.1"
        port = int(sys.argv[4]) if len(sys.argv) >= 5 else 5000
        app = create_web_app(model_path)
        app.run(host=host, port=port, debug=False)
    else:
        data_path = sys.argv[1] if len(sys.argv) >= 2 else default_data_path
        external_path = sys.argv[2] if len(sys.argv) >= 3 else default_external_path
        output_dir = sys.argv[3] if len(sys.argv) >= 4 else default_output_dir

        pipeline = MLPipeline(
            data_path=data_path,
            output_dir=output_dir
        )
        
        pipeline.run_full_pipeline(external_validation_path=external_path)
