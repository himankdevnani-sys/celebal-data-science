from _future_ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "Avg_Price_USD"
FIGDIR = Path("figures")



def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
  
    df["Date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
    return df.sort_values("Date").reset_index(drop=True)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def describe_data(df: pd.DataFrame) -> None:
    print(f"rows={len(df)}  cols={df.shape[1]}")
    print(f"date range: {df['Date'].min():%Y-%m} -> {df['Date'].max():%Y-%m}")
    print("\nnumeric summary:")
    num = df.select_dtypes("number")
    print(num.describe().round(2).T[["mean", "std", "min", "max"]])


def run_eda(df: pd.DataFrame) -> None:
    FIGDIR.mkdir(exist_ok=True)
    num_cols = df.select_dtypes("number").columns

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[TARGET], bins=40, color="#3b7dd8", edgecolor="white")
    ax.set_title("Average price distribution")
    ax.set_xlabel("USD")
    fig.tight_layout()
    fig.savefig(FIGDIR / "price_hist.png", dpi=120)
    plt.close(fig)

  
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(num_cols)), num_cols, rotation=90)
    ax.set_yticks(range(len(num_cols)), num_cols)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Numeric correlations")
    fig.tight_layout()
    fig.savefig(FIGDIR / "corr_heatmap.png", dpi=120)
    plt.close(fig)


    by_model = df.groupby("Model")[TARGET].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    by_model.plot.barh(ax=ax, color="#d8743b")
    ax.set_title("Mean price by model")
    ax.set_xlabel("USD")
    fig.tight_layout()
    fig.savefig(FIGDIR / "price_by_model.png", dpi=120)
    plt.close(fig)

    top = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
    print("\nstrongest correlations with target:")
    print(top.round(3).head())
    print(f"figures written to {FIGDIR}/")

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["years_since_start"] = df["Year"] - df["Year"].min()

    df["prod_gap"] = df["Production_Units"] - df["Estimated_Deliveries"]
    df["co2_per_delivery"] = df["CO2_Saved_tons"] / df["Estimated_Deliveries"]
    return df


def build_preprocessor(num_feats, cat_feats) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), num_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ]
    )


def evaluate(name, y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    out = {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
    }
    print(f"  {name:<16} MAE={out['MAE']:.0f}  RMSE={out['RMSE']:.0f}  R2={out['R2']:.3f}")
    return out


def run_regression(df: pd.DataFrame) -> pd.DataFrame:
    cat_feats = ["Region", "Model", "Source_Type"]
    num_feats = [
        "Estimated_Deliveries", "Production_Units", "Battery_Capacity_kWh",
        "Range_km", "CO2_Saved_tons", "Charging_Stations",
        "month_sin", "month_cos", "years_since_start", "prod_gap",
        "co2_per_delivery",
    ]
    X = df[num_feats + cat_feats]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    pre = build_preprocessor(num_feats, cat_feats)

    results = []


    baseline_pred = np.full(len(y_te), y_tr.mean())
    results.append(evaluate("mean_baseline", y_te, baseline_pred))

    ridge = Pipeline([("pre", pre), ("model", Ridge(random_state=RANDOM_STATE))])
    ridge.fit(X_tr, y_tr)
    results.append(evaluate("ridge", y_te, ridge.predict(X_te)))

    rf = Pipeline(
        [("pre", pre), ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))]
    )
    rf.fit(X_tr, y_tr)
    results.append(evaluate("random_forest", y_te, rf.predict(X_te)))

    param_dist = {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [None, 6, 12, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 1.0],
    }
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=15, cv=4, scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    search.fit(X_tr, y_tr)
    print(f"\n  best RF params: {search.best_params_}")
    results.append(evaluate("rf_tuned", y_te, search.best_estimator_.predict(X_te)))

    return pd.DataFrame(results)


def build_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    ts = (
        df.set_index("Date")[TARGET]
        .resample("MS")
        .mean()
        .to_frame("price")
    )
    for lag in (1, 2, 3, 12):
        ts[f"lag_{lag}"] = ts["price"].shift(lag)
    ts["roll3_mean"] = ts["price"].shift(1).rolling(3).mean()
    ts["roll3_std"] = ts["price"].shift(1).rolling(3).std()
    ts["roll6_mean"] = ts["price"].shift(1).rolling(6).mean()
    return ts.dropna()


def run_forecast(df: pd.DataFrame) -> dict:
    ts = build_monthly_series(df)
    feats = [c for c in ts.columns if c != "price"]
    X, y = ts[feats], ts["price"]

    tscv = TimeSeriesSplit(n_splits=5)
    model = Ridge(random_state=RANDOM_STATE)

    fold_rmse = []
    for tr_idx, te_idx in tscv.split(X):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = model.predict(X.iloc[te_idx])
        fold_rmse.append(np.sqrt(mean_squared_error(y.iloc[te_idx], pred)))


    split = len(ts) - 12
    model.fit(X.iloc[:split], y.iloc[:split])
    pred = model.predict(X.iloc[split:])
    res = evaluate("ridge_ts_holdout", y.iloc[split:], pred)
    print(f"  CV fold RMSE mean={np.mean(fold_rmse):.0f}")

    FIGDIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(y.index, y, label="actual", color="#333")
    ax.plot(y.index[split:], pred, label="forecast", color="#d8743b", lw=2)
    ax.set_title("Monthly mean price: last-12-month forecast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "forecast.png", dpi=120)
    plt.close(fig)
    return res


CSV_NAME = "tesla_deliveries_dataset_2015_2025.csv"


def resolve_data_path() -> str:

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args, _ = parser.parse_known_args()
    if args.data:
        return args.data


    for pattern in (
        f"/kaggle/input/**/{CSV_NAME}",
        f"**/{CSV_NAME}",
    ):
        hits = glob.glob(pattern, recursive=True)
        if hits:
            return hits[0]

    raise FileNotFoundError(
        f"Couldn't find {CSV_NAME}. On Kaggle, attach it via 'Add Input' in the "
        "right-hand panel; locally, place it next to this script or pass --data."
    )


def main() -> None:
    data_path = resolve_data_path()

    print("=== load & clean ===")
    df = basic_clean(load_data(data_path))
    describe_data(df)

    print("\n=== EDA ===")
    run_eda(df)

    print("\n=== regression ===")
    df_feat = add_features(df)
    reg_results = run_regression(df_feat)

    print("\n=== time-series forecast ===")
    run_forecast(df)

    print("\n=== summary ===")
    print(reg_results.round(2).to_string(index=False))


if _name_ == "_main_":
    main()
