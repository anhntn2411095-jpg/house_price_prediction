# ============================================================
# Hanoi Apartment Price Prediction - Chợ Tốt Dataset
# Comprehensive Version with Duplicate Check
#
# Models:
# - Baseline (Dummy Regressor - median)
# - Linear Regression
# - Ridge Regression
# - Random Forest Regressor
#
# Evaluation:
# - MAE
# - RMSE
# - R2
# - Cross-validation (5-fold)
#
# Important:
# - price_per_m2 is NOT used as an input feature.
# - Duplicate and near-duplicate checks are included to avoid inflated performance.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. Display settings
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# ============================================================
# 2. Load dataset
# ============================================================

path = "housing_chotot_rows.csv"

df = pd.read_csv(path)

print("=" * 100)
print("ORIGINAL DATASET")
print("=" * 100)
print("Original dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nPreview:")
print(df.head())

# ============================================================
# 3. Basic filtering and cleaning
# ============================================================

# Keep apartment listings only
if "category" in df.columns:
    df = df[df["category"] == "Căn hộ/Chung cư"].copy()

# Keep sale listings only
if "is_rent" in df.columns:
    df = df[df["is_rent"] == False].copy()

# Remove rows missing essential values
df = df.dropna(subset=["price_billion", "area_m2", "district"])

# Remove invalid values
df = df[
    (df["price_billion"] > 0) &
    (df["area_m2"] > 0)
].copy()

# Create price per m2 in million VND for cleaning only
if "price_per_m2" in df.columns:
    df["price_per_m2_million"] = df["price_per_m2"] / 1_000_000
else:
    df["price_per_m2_million"] = (
        df["price_billion"] * 1_000_000_000 / df["area_m2"] / 1_000_000
    )

# Outlier cleaning
df = df[
    (df["area_m2"] >= 15) &
    (df["area_m2"] <= 300) &
    (df["price_billion"] >= 0.5) &
    (df["price_billion"] <= 100) &
    (df["price_per_m2_million"] >= 10) &
    (df["price_per_m2_million"] <= 500)
].copy()

print("\n" + "=" * 100)
print("AFTER BASIC CLEANING")
print("=" * 100)
print("Cleaned dataset shape:", df.shape)

print("\nPrice summary, billion VND:")
print(df["price_billion"].describe())

print("\nArea summary, m2:")
print(df["area_m2"].describe())

print("\nPrice per m2 summary, million VND/m2:")
print(df["price_per_m2_million"].describe())

# ============================================================
# 3.5 EDA (Exploratory Data Analysis)
# ============================================================

print("\n" + "=" * 100)
print("EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 100)

# Target variable analysis
print("\n1. TARGET VARIABLE (Price) ANALYSIS:")
print(f"   - Skewness: {df['price_billion'].skew():.2f}")
print(f"   - Kurtosis: {df['price_billion'].kurtosis():.2f}")
print(f"   - IQR: {df['price_billion'].quantile(0.75) - df['price_billion'].quantile(0.25):.2f} tỷ")
print("   → Skewness > 1 cho thấy phân phối lệch phải, cần log transformation")

# Create figure for EDA plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Histogram of price
axes[0, 0].hist(df['price_billion'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Price (billion VND)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Apartment Prices')
axes[0, 0].axvline(df['price_billion'].median(), color='red', linestyle='--', label=f'Median: {df["price_billion"].median():.1f}')
axes[0, 0].legend()

# Log-transformed price
axes[0, 1].hist(np.log1p(df['price_billion']), bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Log(Price + 1)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Log-Transformed Price')

# Boxplot of price by district (top 10 districts)
top_districts = df['district'].value_counts().head(10).index
df_top_districts = df[df['district'].isin(top_districts)]
df_top_districts.boxplot(column='price_billion', by='district', ax=axes[0, 2], rot=45)
axes[0, 2].set_title('Price Distribution by District')
axes[0, 2].set_xlabel('')
axes[0, 2].set_ylabel('Price (billion VND)')

# Scatter plot: Price vs Area
axes[1, 0].scatter(df['area_m2'], df['price_billion'], alpha=0.3, s=10)
axes[1, 0].set_xlabel('Area (m2)')
axes[1, 0].set_ylabel('Price (billion VND)')
axes[1, 0].set_title('Price vs Area')
# Add correlation
corr = df['area_m2'].corr(df['price_billion'])
axes[1, 0].text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=axes[1, 0].transAxes)

# Price per m2 by district
district_avg_price_per_m2 = df.groupby('district')['price_per_m2_million'].median().sort_values(ascending=False).head(10)
axes[1, 1].barh(range(len(district_avg_price_per_m2)), district_avg_price_per_m2.values)
axes[1, 1].set_yticks(range(len(district_avg_price_per_m2)))
axes[1, 1].set_yticklabels(district_avg_price_per_m2.index)
axes[1, 1].set_xlabel('Price per m2 (million VND)')
axes[1, 1].set_title('Median Price per m2 by District (Top 10)')

# Correlation heatmap for numeric features
numeric_cols_for_corr = ['price_billion', 'area_m2', 'bedrooms', 'bathrooms', 'floors', 
                          'latitude', 'longitude', 'price_per_m2_million']
numeric_cols_for_corr = [col for col in numeric_cols_for_corr if col in df.columns and df[col].notna().any()]
if len(numeric_cols_for_corr) > 1:
    corr_matrix = df[numeric_cols_for_corr].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 2], square=True)
    axes[1, 2].set_title('Correlation Matrix')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
print("\n2. EDA PLOTS SAVED: eda_plots.png")

# Check missing values percentage
print("\n3. MISSING VALUES ANALYSIS:")
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
if len(missing_pct) > 0:
    print(missing_pct)
else:
    print("   No missing values in any column")

# Check cardinality of categorical features
print("\n4. CATEGORICAL FEATURES CARDINALITY:")
categorical_check = ["district", "ward", "legal_document", "furnishing", "property_status", "seller_type"]
for col in categorical_check:
    if col in df.columns:
        print(f"   {col}: {df[col].nunique()} unique values")

# ============================================================
# 4. Fix data types and missing values
# ============================================================

categorical_columns = [
    "district",
    "ward",
    "legal_document",
    "furnishing",
    "property_status",
    "seller_type"
]

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

for col in keyword_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

numeric_columns = [
    "area_m2",
    "latitude",
    "longitude",
    "bedrooms",
    "bathrooms",
    "floors"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# 5. Duplicate and near-duplicate checking
# ============================================================

print("\n" + "=" * 100)
print("DUPLICATE CHECK")
print("=" * 100)

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

exact_duplicate_cols = [col for col in exact_duplicate_cols if col in df.columns]

exact_duplicate_count = df.duplicated(subset=exact_duplicate_cols).sum()
exact_duplicate_percentage = exact_duplicate_count / len(df) * 100

print("\nExact duplicate columns:")
print(exact_duplicate_cols)

print("\nNumber of exact duplicates:", exact_duplicate_count)
print(f"Exact duplicate percentage: {exact_duplicate_percentage:.2f}%")

# Near-duplicate check excluding price
# This checks if the same-looking apartment appears more than once,
# possibly with the same/different price.
near_duplicate_cols = [
    "area_m2",
    "district",
    "ward",
    "latitude",
    "longitude",
    "bedrooms",
    "bathrooms"
]

near_duplicate_cols = [col for col in near_duplicate_cols if col in df.columns]

near_duplicate_count = df.duplicated(subset=near_duplicate_cols).sum()
near_duplicate_percentage = near_duplicate_count / len(df) * 100

print("\nNear-duplicate columns:")
print(near_duplicate_cols)

print("\nNumber of near duplicates:", near_duplicate_count)
print(f"Near duplicate percentage: {near_duplicate_percentage:.2f}%")

# Save duplicate examples for inspection
if exact_duplicate_count > 0:
    exact_duplicates_df = df[df.duplicated(subset=exact_duplicate_cols, keep=False)].copy()
    exact_duplicates_df.to_csv("exact_duplicate_examples.csv", index=False)
    print("\nSaved exact duplicate examples to: exact_duplicate_examples.csv")

if near_duplicate_count > 0:
    near_duplicates_df = df[df.duplicated(subset=near_duplicate_cols, keep=False)].copy()
    near_duplicates_df.to_csv("near_duplicate_examples.csv", index=False)
    print("Saved near duplicate examples to: near_duplicate_examples.csv")

# Create near-duplicate-removed dataset
df_dedup = df.drop_duplicates(subset=near_duplicate_cols, keep="first").copy()

print("\nDataset size before near-duplicate removal:", df.shape)
print("Dataset size after near-duplicate removal:", df_dedup.shape)
print("Rows removed:", len(df) - len(df_dedup))


# ============================================================
# 6. Define feature sets
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
    "floors",
    "furnishing",
    "property_status",
    "seller_type",
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

enhanced_without_ward_features = [
    "area_m2",
    "district",
    "latitude",
    "longitude",
    "legal_document",
    "bedrooms",
    "bathrooms",
    "floors",
    "furnishing",
    "property_status",
    "seller_type",
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

feature_sets = {
    "Basic Features": basic_features,
    "Enhanced Features": enhanced_features,
    "Enhanced Features Without Ward": enhanced_without_ward_features
}

# ============================================================
# 7. Define models (THÊM BASELINE MODEL)
# ============================================================

models = {
    "Baseline (Dummy - Median)": DummyRegressor(strategy="median"),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(
        alpha=1.0,
        random_state=42
    ),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1
    )
}

# ============================================================
# 8. Preprocessing pipeline (THÊM FEATURE SELECTION VÀ INTERACTION TERMS)
# ============================================================

def get_feature_types(X):
    """
    Define numerical and categorical features safely.
    Avoids Pandas string dtype warnings by using explicit column lists.
    """
    numerical_features = []
    categorical_features = []

    for col in X.columns:
        if col in categorical_columns:
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    return numerical_features, categorical_features


def build_preprocessor(X, use_feature_selection=False, use_interaction=False):
    numerical_features, categorical_features = get_feature_types(X)
    
    # Numerical transformer
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Add interaction terms if requested (chỉ cho numerical features)
    if use_interaction and len(numerical_features) >= 2:
        numerical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("poly", PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
            ("scaler", StandardScaler())
        ])
        print(f"   → Added interaction terms for numerical features: {numerical_features}")
    
    # Categorical transformer
    try:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
    except TypeError:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    
    # Add feature selection if requested
    if use_feature_selection:
        preprocessor = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(f_regression, k=min(20, len(numerical_features) + len(categorical_features) * 10)))
        ])
        print(f"   → Added feature selection (keeping top 20 features)")
    
    return preprocessor, numerical_features, categorical_features


# ============================================================
# 9. Train and evaluate function (THÊM CROSS-VALIDATION)
# ============================================================

def evaluate_models(dataframe, dataset_version, features, feature_set_name, use_log_target=False, use_feature_selection=False, use_interaction=False):
    features = [col for col in features if col in dataframe.columns]

    X = dataframe[features].copy()
    X = X.replace({pd.NA: np.nan})

    y_original = dataframe["price_billion"].copy()

    if use_log_target:
        y = np.log1p(y_original)
        target_type = "Log Price Target"
    else:
        y = y_original
        target_type = "Raw Price Target"

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    if use_log_target:
        y_test_original = np.expm1(y_test)
    else:
        y_test_original = y_test

    preprocessor, numerical_features, categorical_features = build_preprocessor(X_train, use_feature_selection, use_interaction)

    print("\n" + "=" * 120)
    print(f"Running: {dataset_version} | {feature_set_name} | {target_type}")
    if use_feature_selection:
        print("** WITH FEATURE SELECTION **")
    if use_interaction:
        print("** WITH INTERACTION TERMS **")
    print("=" * 120)
    print("Features:", features)
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    results = []
    trained_models = {}
    
    # 5-fold cross-validation settings
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Cross-validation on training set
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)
        
        # Train on full training set
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        if use_log_target:
            y_pred_original = np.expm1(y_pred)
        else:
            y_pred_original = y_pred

        y_pred_original = np.maximum(y_pred_original, 0)

        mae = mean_absolute_error(y_test_original, y_pred_original)
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, y_pred_original)

        results.append({
            "Dataset Version": dataset_version,
            "Feature Set": feature_set_name,
            "Target Type": target_type,
            "Model": model_name,
            "CV R2 Mean (5-fold)": cv_scores.mean(),
            "CV R2 Std (5-fold)": cv_scores.std(),
            "MAE (billion VND)": mae,
            "RMSE (billion VND)": rmse,
            "R2 Score": r2,
            "Number of Features Before Encoding": len(features),
            "Total Rows": len(dataframe),
            "Train Rows": X_train.shape[0],
            "Test Rows": X_test.shape[0]
        })

        trained_models[(dataset_version, feature_set_name, target_type, model_name)] = pipeline

    return results, trained_models


# ============================================================
# 10. Run all experiments (THÊM CÁC THỬ NGHIỆM MỚI)
# ============================================================

# ============================================================
# 10. Run all experiments (RÚT GỌN - chỉ 8 experiments)
# ============================================================

all_results = []
all_trained_models = {}

dataset_versions = {
    "Cleaned Dataset": df,
    "Near-Duplicates Removed": df_dedup
}

# Chỉ dùng 2 feature sets chính
feature_sets_reduced = {
    "Basic Features": basic_features,
    "Enhanced Features": enhanced_features
}

# Chỉ dùng 4 models chính (bỏ Dummy vì đã chạy riêng)
models_reduced = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
}

for dataset_version, current_df in dataset_versions.items():
    for feature_set_name, features in feature_sets_reduced.items():
        features_existing = [col for col in features if col in current_df.columns]
        if len(features_existing) == 0:
            continue
        
        # Chỉ chạy Raw target (bỏ Log)
        results_raw, trained_raw = evaluate_models(
            dataframe=current_df,
            dataset_version=dataset_version,
            features=features_existing,
            feature_set_name=feature_set_name,
            use_log_target=False,  # Cố định là False
            use_feature_selection=False,  # Bỏ
            use_interaction=False  # Bỏ
        )
        all_results.extend(results_raw)
        all_trained_models.update(trained_raw)

# Chạy Baseline riêng (chỉ 1 lần)
print("\n" + "=" * 100)
print("RUNNING BASELINE MODEL")
print("=" * 100)
X_baseline = df[["area_m2"]].copy()  # Dùng 1 feature bất kỳ
y_baseline = df["price_billion"].copy()
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_baseline, y_baseline, test_size=0.2, random_state=42)

dummy = DummyRegressor(strategy="median")
dummy.fit(X_train_b, y_train_b)
y_pred_b = dummy.predict(X_test_b)

baseline_result = {
    "Dataset Version": "Cleaned Dataset",
    "Feature Set": "Baseline",
    "Target Type": "Raw Price Target",
    "Model": "Baseline (Dummy - Median)",
    "CV R2 Mean (5-fold)": 0.0,
    "CV R2 Std (5-fold)": 0.0,
    "MAE (billion VND)": mean_absolute_error(y_test_b, y_pred_b),
    "RMSE (billion VND)": np.sqrt(mean_squared_error(y_test_b, y_pred_b)),
    "R2 Score": r2_score(y_test_b, y_pred_b),
    "Number of Features Before Encoding": 1,
    "Total Rows": len(df),
    "Train Rows": X_train_b.shape[0],
    "Test Rows": X_test_b.shape[0]
}
all_results.append(baseline_result)
# ============================================================
# 11. Final comparison table
# ============================================================

results_df = pd.DataFrame(all_results)

results_df = results_df.sort_values(
    by=["RMSE (billion VND)", "MAE (billion VND)"],
    ascending=True
).reset_index(drop=True)

print("\n" + "=" * 140)
print("FINAL MODEL COMPARISON RESULTS")
print("=" * 140)
print(results_df.to_string())

# ============================================================
# 12. Best model based on lowest RMSE
# ============================================================

best_row = results_df.iloc[0]

best_dataset_version = best_row["Dataset Version"]
best_feature_set = best_row["Feature Set"]
best_target_type = best_row["Target Type"]
best_model_name = best_row["Model"]

print("\n" + "=" * 100)
print("BEST MODEL BASED ON LOWEST RMSE")
print("=" * 100)
print(best_row)

print("\nBest setup:")
print("Dataset version:", best_dataset_version)
print("Feature set:", best_feature_set)
print("Target type:", best_target_type)
print("Model:", best_model_name)


# ============================================================
# 13. Best conservative model
# ============================================================
# For presentation, it is safer to report the best result after near-duplicate removal.
# This helps answer the question: "Is the performance inflated by duplicate listings?"

dedup_results_df = results_df[
    results_df["Dataset Version"] == "Near-Duplicates Removed"
].copy()

if len(dedup_results_df) > 0:
    best_conservative_row = dedup_results_df.iloc[0]

    print("\n" + "=" * 100)
    print("BEST CONSERVATIVE MODEL AFTER NEAR-DUPLICATE REMOVAL")
    print("=" * 100)
    print(best_conservative_row)

    print("\nRecommended result to report in presentation if duplicate count is high:")
    print("Dataset version:", best_conservative_row["Dataset Version"])
    print("Feature set:", best_conservative_row["Feature Set"])
    print("Target type:", best_conservative_row["Target Type"])
    print("Model:", best_conservative_row["Model"])
    print("MAE:", best_conservative_row["MAE (billion VND)"])
    print("RMSE:", best_conservative_row["RMSE (billion VND)"])
    print("R2:", best_conservative_row["R2 Score"])

# ============================================================
# 13.5 Check if Baseline model is beaten (THÊM MỚI)
# ============================================================

print("\n" + "=" * 100)
print("BASELINE COMPARISON CHECK")
print("=" * 100)

baseline_results = results_df[results_df["Model"] == "Baseline (Dummy - Median)"]
best_non_baseline = results_df[results_df["Model"] != "Baseline (Dummy - Median)"].iloc[0] if len(results_df[results_df["Model"] != "Baseline (Dummy - Median)"]) > 0 else None

if len(baseline_results) > 0 and best_non_baseline is not None:
    baseline_r2 = baseline_results.iloc[0]["R2 Score"]
    best_r2 = best_non_baseline["R2 Score"]
    print(f"Baseline R2 Score: {baseline_r2:.4f}")
    print(f"Best model R2 Score: {best_r2:.4f}")
    print(f"Improvement: {(best_r2 - baseline_r2):.4f}")
    if best_r2 > baseline_r2:
        print("✓ Models are learning meaningful patterns (better than baseline)")
    else:
        print("✗ WARNING: Models are NOT learning - performance equals or worse than baseline")

# ============================================================
# 14. Save sample predictions for best model
# ============================================================

if best_dataset_version == "Cleaned Dataset":
    final_df = df.copy()
else:
    final_df = df_dedup.copy()

# Extract base feature set name (remove suffixes like " + Feature Selection", " + Interactions")
best_feature_set_base = best_feature_set.split(" + ")[0] if " + " in best_feature_set else best_feature_set
best_features = feature_sets[best_feature_set_base]
best_features = [col for col in best_features if col in final_df.columns]

X = final_df[best_features].copy()
X = X.replace({pd.NA: np.nan})

y_original = final_df["price_billion"].copy()

use_log_target = best_target_type == "Log Price Target"

if use_log_target:
    y = np.log1p(y_original)
else:
    y = y_original

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

if use_log_target:
    y_test_original = np.expm1(y_test)
else:
    y_test_original = y_test

# Check if best model used feature selection or interactions
use_fs = "Feature Selection" in best_feature_set
use_int = "Interactions" in best_feature_set
preprocessor, numerical_features, categorical_features = build_preprocessor(X_train, use_fs, use_int)

best_model_object = models[best_model_name]

best_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", best_model_object)
])

best_pipeline.fit(X_train, y_train)

y_pred = best_pipeline.predict(X_test)

if use_log_target:
    y_pred_original = np.expm1(y_pred)
else:
    y_pred_original = y_pred

y_pred_original = np.maximum(y_pred_original, 0)

sample_predictions = X_test.copy()
sample_predictions["Actual Price (billion VND)"] = y_test_original.values
sample_predictions["Predicted Price (billion VND)"] = y_pred_original
sample_predictions["Absolute Error (billion VND)"] = (
    sample_predictions["Actual Price (billion VND)"] -
    sample_predictions["Predicted Price (billion VND)"
]).abs()

sample_predictions = sample_predictions.sort_values(
    by="Absolute Error (billion VND)",
    ascending=True
)

print("\n" + "=" * 100)
print("SAMPLE PREDICTIONS FROM BEST MODEL")
print("=" * 100)
print(sample_predictions.head(15))

sample_predictions.to_csv("best_model_sample_predictions_with_duplicate_check.csv", index=False)

print("\nSaved sample predictions to:")
print("best_model_sample_predictions_with_duplicate_check.csv")


# ============================================================
# 15. Random Forest feature importance if best model is Random Forest
# ============================================================

if best_model_name == "Random Forest Regressor":
    print("\n" + "=" * 100)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 100)

    preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
    model_fitted = best_pipeline.named_steps["model"]
    
    # Handle case with feature selection
    if hasattr(preprocessor_fitted, "named_steps") and "feature_selection" in preprocessor_fitted.named_steps:
        # Get feature names before selection
        inner_preprocessor = preprocessor_fitted.named_steps["preprocessor"]
        selector = preprocessor_fitted.named_steps["feature_selection"]
        
        feature_names = []
        numerical_features, categorical_features = get_feature_types(X_train)
        
        # Numerical feature names
        feature_names.extend(numerical_features)
        
        # Categorical one-hot names
        if len(categorical_features) > 0:
            try:
                onehot = inner_preprocessor.named_transformers_["cat"].named_steps["onehot"]
                cat_names = onehot.get_feature_names_out(categorical_features).tolist()
                feature_names.extend(cat_names)
            except Exception:
                pass
        
        # Get selected features
        selected_mask = selector.get_support()
        feature_names = [name for name, selected in zip(feature_names, selected_mask) if selected]
    else:
        feature_names = []
        numerical_features, categorical_features = get_feature_types(X_train)
        
        # Numerical feature names
        feature_names.extend(numerical_features)
        
        # Categorical one-hot names
        if len(categorical_features) > 0:
            try:
                onehot = preprocessor_fitted.named_transformers_["cat"].named_steps["onehot"]
                cat_names = onehot.get_feature_names_out(categorical_features).tolist()
                feature_names.extend(cat_names)
            except Exception:
                pass

    importances = model_fitted.feature_importances_

    min_len = min(len(feature_names), len(importances))

    importance_df = pd.DataFrame({
        "Feature": feature_names[:min_len],
        "Importance": importances[:min_len]
    }).sort_values(by="Importance", ascending=False)

    print(importance_df.head(20))
    
# ============================================================
# 16. Final interpretation helper
# ============================================================

print("\n" + "=" * 100)
print("INTERPRETATION GUIDE")
print("=" * 100)

print("""
1. If the best model on the cleaned dataset is much better than the best model after near-duplicate removal,
   then duplicate listings may be inflating the performance.

2. If the near-duplicate-removed result is still strong, the model performance is more reliable.

3. For presentation, report:
   - exact duplicate count
   - near duplicate count
   - model performance before and after duplicate removal
   - final conservative model result

4. Do not use price_per_m2 as an input feature because it is calculated from the target price.

5. Enhanced features mean additional property-level and text-derived features such as:
   bedrooms, bathrooms, floors, furnishing, legal status, full furniture, lake view, balcony,
   car access, luxury keywords, and similar listing-description signals.
   
6. Cross-validation results (CV R2 Mean) show how stable the model is across different data splits.
   A large standard deviation indicates the model is sensitive to training data.

7. Feature selection helps identify which features actually matter for price prediction.
   If feature selection improves performance, you had irrelevant/noisy features.

8. Interaction terms (area × district, area × bedrooms) capture combined effects.
   For real estate, area × district is often more important than either feature alone.
""")