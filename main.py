# ============================================================
# MAIN PIPELINE: Hanoi Apartment Price Prediction
# ============================================================
# 1. Load the crawled Chợ Tốt / Nhà Tốt apartment dataset
# 2. Clean and preprocess the dataset
# 3. Run EDA and save EDA plots
# 4. Define Basic and Enhanced feature sets
# 5. Train and evaluate three models:
#    - Linear Regression
#    - Ridge Regression
#    - Random Forest Regressor
# 6. Compare models using MAE, RMSE, R², and 5-fold CV R²
# 7. Show Random Forest feature importance
# 8. Run a demonstration prediction
#
# All three models use the SAME cleaned dataset, SAME feature sets,
# SAME train/test split, SAME preprocessing pipeline, and SAME metrics.
# Therefore, the comparison is fair: only the learning algorithm changes.
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

np.random.seed(42)

# ============================================================
# 2. LINEAR REGRESSION MODEL
# ============================================================

class MyLinearRegression(RegressorMixin, BaseEstimator):
    """
    Self-written Linear Regression using the Normal Equation.

    Lecture formula:
        h_theta(x) = theta^T x

    Cost function from Linear Regression lecture:
        E(theta) = (1 / 2m) * sum((h_theta(x_i) - y_i)^2)

    Normal Equation from lecture:
        theta = (X^T X)^(-1) X^T y

    Implementation note:
        We use np.linalg.pinv instead of direct inverse.
        pinv = pseudo-inverse, which is more stable when X^T X is singular
        or nearly singular, especially after one-hot encoding categorical data.
    """

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Convert input to numpy arrays to support Scikit-learn Pipeline output.
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples = X.shape[0]

        # Add a bias/intercept column of 1s.
        # X_b = [1, x1, x2, ..., xn]
        X_b = np.hstack([np.ones((n_samples, 1)), X])

        # Normal equation:
        # theta = (X^T X)^+ X^T y
        # where + means pseudo-inverse.
        weights_full = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = weights_full[0]
        self.weights = weights_full[1:]
        # Scikit-learn compatibility: fitted attributes end with underscore.
        self.bias_ = self.bias
        self.weights_ = self.weights
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights_ + self.bias_


# ============================================================
# 3. RIDGE REGRESSION MODEL
# ============================================================

class MyRidgeRegression(RegressorMixin, BaseEstimator):
    """
    Ridge Regression using the regularized Normal Equation.

    Ridge Regression is Linear Regression + L2 regularization.

    Regularization lecture idea:
        Regularization reduces the magnitude of coefficients theta_j
        to make the model less prone to overfitting.

    Regularized cost function:
        E(theta) = (1 / 2m) * sum((h_theta(x_i) - y_i)^2)
                   + lambda * sum(theta_j^2)

    Regularized Normal Equation from lecture:
        theta = (X^T X + lambda * I)^(-1) X^T y

    Implementation note:
        I[0, 0] = 0 so that the intercept/bias term is NOT regularized.
        This follows the standard convention: regularize feature weights,
        not the intercept.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.weights_full = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        n_samples, n_features = X.shape

        # Add bias/intercept column.
        X_b = np.hstack([np.ones((n_samples, 1)), X])

        # Identity matrix for L2 regularization.
        I = np.eye(X_b.shape[1])

        # Do not regularize the intercept term.
        I[0, 0] = 0

        # Ridge closed-form solution:
        # theta = (X^T X + alpha * I)^(-1) X^T y
        self.weights_full = np.linalg.solve(
            X_b.T @ X_b + self.alpha * I,
            X_b.T @ y
        )

        self.bias = self.weights_full[0]
        self.weights = self.weights_full[1:]
        # Scikit-learn compatibility: fitted attributes end with underscore.
        self.bias_ = self.bias
        self.weights_ = self.weights
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.weights_ + self.bias_


# ============================================================
# 4. LOAD DATASET
# ============================================================

path = "housing_chotot_rows.csv"

df = pd.read_csv(path)

print("Original dataset shape:", df.shape)


# ============================================================
# 5. DATA CLEANING AND FILTERING
# ============================================================

# Keep apartment listings only.
# Project scope: Hanoi apartment / condominium sale price prediction.
if "category" in df.columns:
    df = df[df["category"] == "Căn hộ/Chung cư"].copy()

# Keep sale listings only, not rental listings.
# Rent price and sale price are different prediction tasks.
if "is_rent" in df.columns:
    df = df[df["is_rent"] == False].copy()

# Remove rows missing essential values.
# price_billion = target, area_m2 and district = key predictors.
df = df.dropna(subset=["price_billion", "area_m2", "district"])

# Remove invalid values.
# Price and area must be positive.
df = df[
    (df["price_billion"] > 0) &
    (df["area_m2"] > 0)
].copy()

# Create price per m2 in million VND for cleaning only.
# Important: price_per_m2 is NOT used as a model input because it is
# calculated from the target price and area, so it would cause data leakage.
if "price_per_m2" in df.columns:
    df["price_per_m2_million"] = df["price_per_m2"] / 1_000_000
else:
    df["price_per_m2_million"] = (
        df["price_billion"] * 1_000_000_000 / df["area_m2"] / 1_000_000
    )

# Outlier cleaning.
# These are rule-based thresholds to remove unrealistic listing records.
df = df[
    (df["area_m2"] >= 15) &
    (df["area_m2"] <= 300) &
    (df["price_billion"] >= 0.5) &
    (df["price_billion"] <= 100) &
    (df["price_per_m2_million"] >= 10) &
    (df["price_per_m2_million"] <= 500)
].copy()

# Remove exact duplicate listings.
# Exact duplicates are repeated rows with the same core apartment information.
exact_duplicate_cols = [
    "area_m2",
    "district",
    "ward",
    "latitude",
    "longitude",
    "bedrooms",
    "bathrooms",
    "price_billion"
]

# Keep only duplicate columns that actually exist in the dataset.
exact_duplicate_cols = [col for col in exact_duplicate_cols if col in df.columns]

before_exact_duplicate_removal = len(df)

df = df.drop_duplicates(subset=exact_duplicate_cols, keep="first").copy()

after_exact_duplicate_removal = len(df)

print("Exact duplicate columns:", exact_duplicate_cols)
print("Exact duplicates removed:", before_exact_duplicate_removal - after_exact_duplicate_removal)
print("Cleaned dataset shape after exact duplicate removal:", df.shape)


# ============================================================
# 6. EDA: EXPLORATORY DATA ANALYSIS
# ============================================================
# EDA is conducted on the cleaned modeling dataset so that the statistics
# and plots match the data actually used for model training.

# Log-transformations are used for visualization because apartment prices
# are right-skewed: most listings are lower/middle price, while a smaller
# number of expensive listings creates a long right tail.
df["price_log"] = np.log1p(df["price_billion"])
df["area_log"] = np.log1p(df["area_m2"])

fig, axes = plt.subplots(4, 2, figsize=(16, 24))

sns.histplot(df["price_billion"], bins=40, kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Original Price Distribution")
axes[0, 0].set_xlabel("Price (billion VND)")

sns.histplot(df["price_log"], bins=40, kde=True, ax=axes[0, 1], color="steelblue")
axes[0, 1].set_title("Log-Transformed Price Distribution")
axes[0, 1].set_xlabel("log(Price)")

sns.histplot(df["area_m2"], bins=40, kde=True, ax=axes[1, 0], color="steelblue")
axes[1, 0].set_title("Original Area Distribution")
axes[1, 0].set_xlabel("Area (m²)")

sns.histplot(df["area_log"], bins=40, kde=True, ax=axes[1, 1], color="steelblue")
axes[1, 1].set_title("Log-Transformed Area Distribution")
axes[1, 1].set_xlabel("log(Area)")

sns.boxplot(y=df["price_billion"], ax=axes[2, 0])
axes[2, 0].set_title("Price Boxplot")

sns.boxplot(y=df["area_m2"], ax=axes[2, 1])
axes[2, 1].set_title("Area Boxplot")

sns.scatterplot(
    x=df["area_m2"],
    y=df["price_billion"],
    alpha=0.5,
    ax=axes[3, 0]
)
axes[3, 0].set_title("Area vs Price")
axes[3, 0].set_xlabel("Area (m²)")
axes[3, 0].set_ylabel("Price (billion VND)")

numeric_cols = [
    "price_billion",
    "area_m2",
    "bedrooms",
    "bathrooms",
    "price_per_m2_million"
]

numeric_cols = [col for col in numeric_cols if col in df.columns]

if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=axes[3, 1]
    )
    axes[3, 1].set_title("Correlation Matrix")

plt.tight_layout()
plt.savefig("eda_combined.png", dpi=200, bbox_inches="tight")
plt.close()

# Numerical EDA summary for slide/report use.
eda_summary_cols = ["price_billion", "area_m2", "price_per_m2_million"]
eda_summary_cols = [col for col in eda_summary_cols if col in df.columns]
print("\nEDA summary:")
print(df[eda_summary_cols].agg(["mean", "median", "min", "max"]).T)

if "price_billion" in df.columns:
    print("\nPrice skewness:", df["price_billion"].skew())
    print("Price kurtosis:", df["price_billion"].kurtosis())

# Check for missing values after cleaning.
missing_values = df.isnull().sum()
missing_percent = 100 * missing_values / len(df)
missing_table = pd.DataFrame({"Missing Count": missing_values, "Missing %": missing_percent})
missing_table = missing_table[missing_table["Missing Count"] > 0].sort_values("Missing %", ascending=False)

if not missing_table.empty:
    print("\nMissing values after cleaning:")
    print(missing_table)
else:
    print("\nNo missing values found after cleaning.")

# Categorical feature analysis.
categorical_cols_for_eda = ["district", "legal_document", "furnishing", "property_status"]

for col in categorical_cols_for_eda:
    if col in df.columns:
        print(f"\nTop 10 most frequent values in '{col}':")
        print(df[col].value_counts().head(10))

        avg_price_cat = df.groupby(col)["price_billion"].mean().sort_values(ascending=False).head(10)
        print(f"\nAverage price by '{col}' (top 10):")
        print(avg_price_cat)


# ============================================================
# 7. DATA TYPE FIXING BEFORE MODELING
# ============================================================

categorical_columns = [
    "district",
    "ward",
    "legal_document",
    "furnishing",
    "property_status"
]

# Categorical missing values are represented as "Unknown".
for col in categorical_columns:
    if col in df.columns:
        df[col] = df[col].astype("object")
        df[col] = df[col].where(pd.notna(df[col]), "Unknown")
        df[col] = df[col].astype(str)

keyword_columns = [
    "has_full_furniture",
    "has_balcony",
    "has_lake_view",
    "has_city_view",
    "has_red_book",
    "has_car_access",
    "has_luxury_keyword",
    "has_new_keyword",
    "has_corner_keyword",
    "has_near_center"
]

# Binary keyword features:
# 1 = keyword appears in title/description
# 0 = keyword does not appear
for col in keyword_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

numeric_columns = [
    "area_m2",
    "latitude",
    "longitude",
    "bedrooms",
    "bathrooms"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 8. DEFINE FEATURE SETS
# ============================================================

basic_features = [
    "area_m2",
    "district",
    "ward",
    "latitude",
    "longitude",
    "legal_document"
]

enhanced_features = [
    "area_m2",
    "district",
    "ward",
    "latitude",
    "longitude",
    "legal_document",
    "bedrooms",
    "bathrooms",
    "furnishing",
    "property_status",
    "has_full_furniture",
    "has_balcony",
    "has_lake_view",
    "has_city_view",
    "has_red_book",
    "has_car_access",
    "has_luxury_keyword",
    "has_new_keyword",
    "has_corner_keyword",
    "has_near_center"
]


# ============================================================
# 9. DEFINE MODELS
# ============================================================
# Linear and Ridge are self-written.
# Random Forest is kept from Scikit-learn as a nonlinear benchmark.

models = {
    "Self-written Linear Regression": MyLinearRegression(),
    "Self-written Ridge Regression": MyRidgeRegression(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 10. PREPROCESSING PIPELINE
# ============================================================

def get_feature_types(X):
    """
    Split selected features into numerical and categorical groups.

    Numerical features:
        median imputation + StandardScaler

    Categorical features:
        fill missing value with 'Unknown' + one-hot encoding
    """
    numerical_features = []
    categorical_features = []

    for col in X.columns:
        if col in categorical_columns:
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    return numerical_features, categorical_features


def build_preprocessor(X):
    """
    Build a Scikit-learn ColumnTransformer.

    Why use Pipeline/ColumnTransformer?
    - The same preprocessing is applied in training, cross-validation,
      test evaluation, and demo prediction.
    - The transformers are fitted only on the training fold inside CV,
      reducing data leakage risk.
    """
    numerical_features, categorical_features = get_feature_types(X)

    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numerical_features, categorical_features


# ============================================================
# 11. TRAIN AND EVALUATE MODELS
# ============================================================

def evaluate_models(dataframe, features, feature_set_name):
    """
    Train and evaluate all models on one selected feature set.

    Target transformation:
        y = log1p(price_billion)

    Reason:
        Apartment prices are right-skewed. log1p compresses large prices
        and makes the target easier for the model to learn.

    After prediction:
        expm1() converts the prediction back to billion VND.

    Evaluation metrics:
        MAE  = average absolute error
        RMSE = square-root of MSE, penalizes large errors more strongly
        R²   = proportion of price variance explained by the model
    """

    features = [col for col in features if col in dataframe.columns]

    X = dataframe[features].copy()
    X = X.replace({pd.NA: np.nan})

    # Use log-transformed target for model training.
    y = np.log1p(dataframe["price_billion"].copy())

    # 80/20 train-test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor, numerical_features, categorical_features = build_preprocessor(X_train)

    print("\n" + "=" * 100)
    print(f"FEATURE SET: {feature_set_name}")
    print("=" * 100)
    print("Features:", features)
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # 5-fold Cross-Validation, matching the model evaluation lecture idea:
    # split training data into folds, try each fold as validation, average results.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Cross-validation R² is calculated on the log target because the
        # model is trained on log1p(price_billion).
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="r2",
            n_jobs=1
        )

        # Train on full training set.
        pipeline.fit(X_train, y_train)

        # Predict log price, then convert back to original billion VND scale.
        y_pred_log = pipeline.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_real = np.expm1(y_test)

        mae = mean_absolute_error(y_test_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))
        r2 = r2_score(y_test_real, y_pred)

        results.append({
            "Feature Set": feature_set_name,
            "Model": model_name,
            "CV R2 Mean": cv_scores.mean(),
            "CV R2 Std": cv_scores.std(),
            "MAE (billion VND)": round(mae, 3),
            "RMSE (billion VND)": round(rmse, 3),
            "R2 Score": round(r2, 4)
        })

        print(f"\n{model_name}:")
        print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  MAE: {mae:.3f} billion VND")
        print(f"  RMSE: {rmse:.3f} billion VND")
        print(f"  R²: {r2:.4f}")

    return results


# ============================================================
# 12. RUN EXPERIMENTS
# ============================================================

all_results = []

results_basic = evaluate_models(df, basic_features, "Basic Features (6 features)")
all_results.extend(results_basic)

results_enhanced = evaluate_models(df, enhanced_features, "Enhanced Features (20 features)")
all_results.extend(results_enhanced)


# ============================================================
# 13. FINAL MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 100)
print("FINAL COMPARISON TABLE")
print("=" * 100)
print(results_df.to_string(index=False))

best_row = results_df.loc[results_df["R2 Score"].idxmax()]

print("\n" + "=" * 100)
print("BEST MODEL")
print("=" * 100)
print(f"Feature Set: {best_row['Feature Set']}")
print(f"Model: {best_row['Model']}")
print(f"R² Score: {best_row['R2 Score']}")
print(f"MAE: {best_row['MAE (billion VND)']} billion VND")
print(f"RMSE: {best_row['RMSE (billion VND)']} billion VND")
print(f"CV R²: {best_row['CV R2 Mean']:.4f} ± {best_row['CV R2 Std']:.4f}")


# ============================================================
# 14. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================
# Feature importance is extracted from the Random Forest model because
# Linear/Ridge coefficients are harder to compare directly after one-hot
# encoding and scaling. Random Forest importance gives an interpretable
# ranking of useful predictors.

print("\nRANDOM FOREST FEATURE IMPORTANCE")

features_enhanced = [col for col in enhanced_features if col in df.columns]
X_enhanced = df[features_enhanced].copy()
X_enhanced = X_enhanced.replace({pd.NA: np.nan})

# This follows the original project code: use raw price for interpretation.
y_importance = df["price_billion"].copy()

preprocessor, num_features, cat_features = build_preprocessor(X_enhanced)

rf_final = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

rf_final.fit(X_enhanced, y_importance)

preprocessor_fitted = rf_final.named_steps["preprocessor"]
model_fitted = rf_final.named_steps["model"]

feature_names = []
feature_names.extend(num_features)

if len(cat_features) > 0:
    try:
        onehot = preprocessor_fitted.named_transformers_["cat"].named_steps["onehot"]
        cat_names = onehot.get_feature_names_out(cat_features).tolist()
        feature_names.extend(cat_names)
    except Exception:
        pass

importances = model_fitted.feature_importances_
min_len = min(len(feature_names), len(importances))

importance_df = pd.DataFrame({
    "Feature": feature_names[:min_len],
    "Importance": importances[:min_len]
}).sort_values(by="Importance", ascending=False)

print(importance_df.head(15).to_string(index=False))


# ============================================================
# 15. DEMONSTRATION PREDICTION
# ============================================================
# Demo model follows the original project code: train Random Forest on all
# cleaned data using Enhanced Features, then predict a user-provided case.

print("\nDEMONSTRATION")

X_demo_train = df[features_enhanced].copy()
X_demo_train = X_demo_train.replace({pd.NA: np.nan})
y_demo_train = df["price_billion"].copy()

demo_preprocessor, _, _ = build_preprocessor(X_demo_train)

demo_model = Pipeline(steps=[
    ("preprocessor", demo_preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

demo_model.fit(X_demo_train, y_demo_train)

case_1 = pd.DataFrame([{
    "area_m2": 75,
    "district": "Quận Cầu Giấy",
    "ward": "Phường Dịch Vọng",
    "latitude": 21.0368,
    "longitude": 105.7900,
    "bedrooms": 2,
    "bathrooms": 2,
}])


def predict_apartment(input_df):
    """
    Predict apartment price from user-provided listing features.

    The function:
    1. Prints the original input
    2. Removes columns not used by the model
    3. Adds missing Enhanced Feature columns as NaN
    4. Reorders columns to match training feature order
    5. Predicts price in billion VND
    """

    print("\nOriginal Input:")
    print(input_df.to_string(index=False))

    # Remove extra columns not used by the model.
    input_df = input_df[
        [col for col in input_df.columns if col in features_enhanced]
    ].copy()

    # Add missing columns required by Enhanced Features.
    for col in features_enhanced:
        if col not in input_df.columns:
            input_df[col] = np.nan

    # Reorder input columns to match model training order.
    input_df = input_df[features_enhanced]

    prediction = demo_model.predict(input_df)[0]

    print(f"\nPredicted Price: {prediction:.2f} billion VND")


predict_apartment(case_1)
