import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_b = np.hstack([np.ones((n_samples, 1)), X])
        I = np.eye(X_b.shape[1])
        I[0, 0] = 0
        self.weights_full = np.linalg.solve(
            X_b.T @ X_b + self.alpha * I,
            X_b.T @ y
        )
        self.bias    = self.weights_full[0]
        self.weights = self.weights_full[1:]

    def predict(self, X):
        return X @ self.weights + self.bias

basic_features = [
    "area_m2", "district", "ward", "latitude", "longitude", "legal_document"
]

enhanced_features = [
    "area_m2", "district", "ward", "latitude", "longitude", "legal_document",
    "bedrooms", "bathrooms", "furnishing", "property_status",
    "has_full_furniture", "has_balcony", "has_lake_view", "has_city_view",
    "has_red_book", "has_car_access", "has_luxury_keyword", "has_new_keyword",
    "has_corner_keyword", "has_near_center"
]

NUMERIC_COLS     = ["area_m2", "latitude", "longitude", "bedrooms", "bathrooms"]
CATEGORICAL_COLS = ["ward", "legal_document", "district", "furnishing", "property_status"]
BINARY_COLS      = [
    "has_full_furniture", "has_balcony", "has_lake_view", "has_city_view",
    "has_red_book", "has_car_access", "has_luxury_keyword", "has_new_keyword",
    "has_corner_keyword", "has_near_center"
]

def prepare_features(df, feature_set,
                     num_imputer=None, encoder=None,
                     scaler_mean=None, scaler_std=None, fit=True):

    num_cols = [c for c in NUMERIC_COLS     if c in feature_set]
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_set]
    bin_cols = [c for c in BINARY_COLS      if c in feature_set]

    parts = []

    # numeric
    if num_cols:
        if fit:
            num_imputer = SimpleImputer(strategy="median")
            X_num = num_imputer.fit_transform(df[num_cols].astype(float))
        else:
            X_num = num_imputer.transform(df[num_cols].astype(float))
        parts.append(X_num)

    # categorical
    if cat_cols:
        if fit:
            encoder = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            X_cat = encoder.fit_transform(df[cat_cols].astype(str))
        else:
            X_cat = encoder.transform(df[cat_cols].astype(str))
        parts.append(X_cat)

    # binary
    if bin_cols:
        parts.append(df[bin_cols].values.astype(float))

    X = np.hstack(parts)

    # standardize
    if fit:
        scaler_mean = X.mean(axis=0)
        scaler_std  = X.std(axis=0)
        scaler_std[scaler_std == 0] = 1
    X = (X - scaler_mean) / scaler_std

    return X, num_imputer, encoder, scaler_mean, scaler_std

def get_feature_names(feature_set, encoder):
    num_cols = [c for c in NUMERIC_COLS     if c in feature_set]
    cat_cols = [c for c in CATEGORICAL_COLS if c in feature_set]
    bin_cols = [c for c in BINARY_COLS      if c in feature_set]
    cat_names = encoder.named_steps["onehot"].get_feature_names_out(cat_cols).tolist() if cat_cols else []
    return num_cols + cat_names + bin_cols

df = pd.read_csv("housing_chotot_rows.csv")

all_needed = list(set(enhanced_features + ["price_billion"]))
df = df[all_needed].dropna(subset=["price_billion"])

p99 = df["price_billion"].quantile(0.99)
df  = df[df["price_billion"] <= p99].reset_index(drop=True)
print(f"Rows after clipping outliers (p99={p99:.2f}): {len(df)}")

y_raw = df["price_billion"].values.astype(float)
y     = np.log1p(y_raw)

train_df, test_df, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42)
y_test_raw = np.expm1(y_test)

def run_experiment(name, feature_set):
    print(f"\nPreparing features: {name}...")
    X_train, num_imp, enc, s_mean, s_std = prepare_features(
        train_df, feature_set, fit=True)
    X_test, _, _, _, _ = prepare_features(
        test_df, feature_set,
        num_imputer=num_imp, encoder=enc,
        scaler_mean=s_mean, scaler_std=s_std, fit=False)

    ridge = RidgeRegression(alpha=1.0)
    ridge.fit(X_train, y_train)

    y_pred_log = ridge.predict(X_test)
    y_pred_raw = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred_raw))
    mae  = mean_absolute_error(y_test_raw, y_pred_raw)
    r2   = r2_score(y_test_raw, y_pred_raw)

    print(f"\n{'=' * 45}")
    print(f"  {name.upper()}")
    print(f"{'=' * 45}")
    print(f"  Features : {len(feature_set)}")
    print(f"  R²       : {r2:.4f}")
    print(f"  RMSE     : {rmse:.4f} billion VND")
    print(f"  MAE      : {mae:.4f} billion VND")

    return r2, rmse, mae

r2_basic,    rmse_basic,    mae_basic    = run_experiment("Basic features",    basic_features)
r2_enhanced, rmse_enhanced, mae_enhanced = run_experiment("Enhanced features", enhanced_features)