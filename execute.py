import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

path = "housing_chotot_rows.csv"

df = pd.read_csv(path)

print("Original dataset shape:", df.shape)

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

# Remove exact duplicate listings 
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

before_exact_duplicate_removal = len(df)

df = df.drop_duplicates(subset=exact_duplicate_cols, keep="first").copy()

after_exact_duplicate_removal = len(df)

print("Exact duplicate columns:", exact_duplicate_cols)
print("Cleaned dataset shape after exact duplicate removal:", df.shape)


df["price_log"] = np.log1p(df["price_billion"])
df["area_log"] = np.log1p(df["area_m2"])

fig, axes = plt.subplots(4, 2, figsize=(16, 24))

sns.histplot(df["price_billion"], bins=40, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("Original Price Distribution")
axes[0, 0].set_xlabel("Price (billion VND)")

sns.histplot(df["price_log"], bins=40, kde=True, ax=axes[0, 1])
axes[0, 1].set_title("Log-Transformed Price Distribution")
axes[0, 1].set_xlabel("log(Price)")

sns.histplot(df["area_m2"], bins=40, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Original Area Distribution")
axes[1, 0].set_xlabel("Area (m²)")

sns.histplot(df["area_log"], bins=40, kde=True, ax=axes[1, 1])
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

# Check for missing values after cleaning
missing_values = df.isnull().sum()
missing_percent = 100 * missing_values / len(df)
missing_table = pd.DataFrame({"Missing Count": missing_values, "Missing %": missing_percent})
missing_table = missing_table[missing_table["Missing Count"] > 0].sort_values("Missing %", ascending=False)
if not missing_table.empty:
    print("\nMissing values after cleaning:")
    print(missing_table)
else:
    print("\nNo missing values found after cleaning.")

# Categorical feature analysis
categorical_cols = ["district", "legal_document", "furnishing", "property_status"]
for col in categorical_cols:
    if col in df.columns:
        print(f"\nTop 10 most frequent values in '{col}':")
        print(df[col].value_counts().head(10))
        
        # Average price per category
        avg_price_cat = df.groupby(col)["price_billion"].mean().sort_values(ascending=False).head(10)
        print(f"\nAverage price by '{col}' (top 10):")
        print(avg_price_cat)

categorical_columns = [
    "district",
    "ward",
    "legal_document",
    "furnishing",
    "property_status"
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
    "bathrooms"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


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


models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
}


def get_feature_types(X):
    numerical_features = []
    categorical_features = []

    for col in X.columns:
        if col in categorical_columns:
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    return numerical_features, categorical_features


def build_preprocessor(X):
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


#Train and evaluate function

def evaluate_models(dataframe, features, feature_set_name):
    features = [col for col in features if col in dataframe.columns]

    X = dataframe[features].copy()
    X = X.replace({pd.NA: np.nan})

    y = np.log1p(dataframe["price_billion"].copy())

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

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for model_name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='r2', n_jobs=-1)

        # Train
        pipeline.fit(X_train, y_train)
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

#Run experiments

all_results = []

# Basic Features
results_basic = evaluate_models(df, basic_features, "Basic Features (6 features)")
all_results.extend(results_basic)

# Enhanced Features
results_enhanced = evaluate_models(df, enhanced_features, "Enhanced Features (22 features)")
all_results.extend(results_enhanced)


#Final comparison table

results_df = pd.DataFrame(all_results)

print(results_df.to_string(index=False))

#Best model

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

#Feature importance from Random Forest
print("RANDOM FOREST FEATURE IMPORTANCE")

# Train final Random Forest on full enhanced features
features_enhanced = [col for col in enhanced_features if col in df.columns]
X_enhanced = df[features_enhanced].copy()
X_enhanced = X_enhanced.replace({pd.NA: np.nan})
y = df["price_billion"].copy()

preprocessor, num_features, cat_features = build_preprocessor(X_enhanced)

rf_final = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))
])

rf_final.fit(X_enhanced, y)

# Get feature names after preprocessing
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

# Demonstration 

print("DEMONSTRATION")

features_enhanced = [col for col in enhanced_features if col in df.columns]

X_enhanced = df[features_enhanced].copy()
X_enhanced = X_enhanced.replace({pd.NA: np.nan})

y = df["price_billion"].copy()

demo_preprocessor, _, _ = build_preprocessor(X_enhanced)

demo_model = Pipeline(steps=[
    ("preprocessor", demo_preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ))
])

demo_model.fit(X_enhanced, y)

case_1 = pd.DataFrame([{
    "area_m2": 75,
    "district": "Cầu Giấy",
    "ward": "Dịch Vọng",
    "latitude": 21.0368,
    "longitude": 105.7900,
    "bedrooms": 2,
    "bathrooms": 2,
}])

def predict_apartment(input_df):

    print("\nOriginal Input:")
    print(input_df.to_string(index=False))

    # Remove extra columns
    input_df = input_df[
        [col for col in input_df.columns if col in features_enhanced]
    ]

    # Add missing columns
    for col in features_enhanced:
        if col not in input_df.columns:
            input_df[col] = np.nan

    input_df = input_df[features_enhanced]

    prediction = demo_model.predict(input_df)[0]

    print(f"\nPredicted Price: {prediction:.2f} billion VND")

predict_apartment(case_1)
