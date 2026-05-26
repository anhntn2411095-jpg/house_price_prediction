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

print("Cleaned dataset shape:", df.shape)

#Fix data types and missing values

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

#Define feature sets

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

#Define models

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
}

#Preprocessing pipeline

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

    y = dataframe["price_billion"].copy()

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
        y_pred = pipeline.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

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
        print(f"  MAE: {mae:.3f} tỷ VND")
        print(f"  RMSE: {rmse:.3f} tỷ VND")
        print(f"  R²: {r2:.4f}")

    return results


#Run experiments

all_results = []

print("\n" + "#" * 100)
print("# RUNNING EXPERIMENTS ON CLEANED DATASET")
print("#" * 100)

# Basic Features
results_basic = evaluate_models(df, basic_features, "Basic Features (6 features)")
all_results.extend(results_basic)

# Enhanced Features
results_enhanced = evaluate_models(df, enhanced_features, "Enhanced Features (22 features)")
all_results.extend(results_enhanced)


#Final comparison table

results_df = pd.DataFrame(all_results)

print("\n" + "=" * 100)
print("MODEL COMPARISON")
print("=" * 100)
print(results_df.to_string(index=False))

best_row = results_df.loc[results_df["R2 Score"].idxmax()]

print("\n" + "=" * 100)
print("BEST MODEL")
print("=" * 100)
print(f"Feature Set: {best_row['Feature Set']}")
print(f"Model: {best_row['Model']}")
print(f"R² Score: {best_row['R2 Score']}")
print(f"MAE: {best_row['MAE (billion VND)']} tỷ VND")
print(f"RMSE: {best_row['RMSE (billion VND)']} tỷ VND")
print(f"CV R²: {best_row['CV R2 Mean']:.4f} ± {best_row['CV R2 Std']:.4f}")

#Feature importance from Random Forest
print("\n" + "=" * 100)
print("RANDOM FOREST FEATURE IMPORTANCE (Enhanced Features)")
print("=" * 100)

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

#Demonstration

print("\n" + "=" * 100)
print("DEMONSTRATION - INPUT CASES")
print("=" * 100)

# Train final model
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

    # Reorder columns
    input_df = input_df[features_enhanced]

    prediction = demo_model.predict(input_df)[0]

    print(f"\nPredicted Price: {prediction:.2f} billion VND")

predict_apartment(case_1)
