# -*- coding: utf-8 -*-
"""
data_preparation.py — Olist E-Commerce Data Preparation

Loads 9 raw CSVs from the Olist Brazilian E-Commerce Public Dataset,
merges into an order-level master dataset, computes haversine shipping
distances, engineers features, and applies a chronological 80/20
train/test split to prevent temporal leakage.

Usage:
    python data_preparation.py [--data-dir path/to/raw/csvs]

Dataset: Olist Brazilian E-Commerce (Kaggle, CC BY-NC-SA 4.0).
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Configuration ────────────────────────────────────────────────────
# Relative paths are dynamically resolved from __file__ to ensure the 
# script reliably executes whether located locally, on OneDrive, or Google Drive.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Defaulting raw data loads to a relative 'data/raw' folder 
RAW_DIR = os.path.join(SCRIPT_DIR, "data", "raw")
# Defaulting clean exports to a relative 'data/processed' folder
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "processed")
# Ensure the output directory structure exists before attempting to save
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Haversine ────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two coordinate arrays."""
    R = 6_371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ── 1. Load ──────────────────────────────────────────────────────────

def load_raw_data(data_dir):
    """Load all 9 Olist CSV files and return as a dict of DataFrames."""
    print("1  Loading raw data")
    files = {
        "orders":      "olist_orders_dataset.csv",
        "reviews":     "olist_order_reviews_dataset.csv",
        "items":       "olist_order_items_dataset.csv",
        "payments":    "olist_order_payments_dataset.csv",
        "products":    "olist_products_dataset.csv",
        "customers":   "olist_customers_dataset.csv",
        "sellers":     "olist_sellers_dataset.csv",
        "geolocation": "olist_geolocation_dataset.csv",
        "categories":  "product_category_name_translation.csv",
    }
    data = {}
    for key, fname in files.items():
        path = os.path.join(data_dir, fname)
        if not os.path.isfile(path):
            sys.exit(f"ERROR: {path} not found. Use --data-dir to specify location.")
        data[key] = pd.read_csv(path)
        print(f"   {key:<15} {data[key].shape[0]:>10,} rows × {data[key].shape[1]} cols")
    return data


# ── 2. Clean & Merge ─────────────────────────────────────────────────

def clean_and_merge(data):
    """
    Build order-level master dataset:
      1. Filter to delivered orders with actual delivery dates.
      2. Merge reviews, items, payments, products, customers, sellers.
      3. Integrate geolocation and compute haversine distance.
    """
    print("\n2  Cleaning and merging")

    # Filter to purely delivered orders for accurate logistics study
    # Delivered orders only
    orders = data["orders"].copy()
    orders = orders[orders["order_status"] == "delivered"].copy()
    dt_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in dt_cols:
        orders[col] = pd.to_datetime(orders[col])
    orders = orders.dropna(subset=["order_delivered_customer_date"])
    print(f"   Delivered orders with dates: {len(orders):,}")

    # Reviews
    reviews = data["reviews"][["order_id", "review_score"]].copy()
    df = orders.merge(reviews, on="order_id", how="inner")

    # Items ─ aggregate per order
    items = data["items"].copy()
    items_agg = items.groupby("order_id").agg(
        seller_id=("seller_id", "first"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        num_items=("order_item_id", "count"),
    ).reset_index()
    items_agg["total_order_value"] = items_agg["total_price"] + items_agg["total_freight"]
    items_agg["freight_ratio"] = (
        items_agg["total_freight"] / items_agg["total_order_value"]
    ).fillna(0)
    df = df.merge(items_agg, on="order_id", how="left")

    # Payments ─ aggregate per order
    payments = data["payments"].copy()
    pay_agg = payments.groupby("order_id").agg(
        payment_installments=("payment_installments", "max"),
        primary_payment_type=(
            "payment_type",
            lambda x: x.mode().iloc[0] if len(x) > 0 else "unknown",
        ),
    ).reset_index()
    df = df.merge(pay_agg, on="order_id", how="left")

    # Products ─ primary product per order
    products = data["products"].merge(
        data["categories"], on="product_category_name", how="left"
    )
    item_prods = items[["order_id", "product_id"]].drop_duplicates("order_id")
    item_prods = item_prods.merge(
        products[["product_id", "product_category_name_english", "product_weight_g"]],
        on="product_id", how="left",
    )
    df = df.merge(
        item_prods[["order_id", "product_category_name_english", "product_weight_g"]],
        on="order_id", how="left",
    )

    # Customer & seller locations
    df = df.merge(
        data["customers"][["customer_id", "customer_state", "customer_zip_code_prefix"]],
        on="customer_id", how="left",
    )
    df = df.merge(
        data["sellers"][["seller_id", "seller_state", "seller_zip_code_prefix"]],
        on="seller_id", how="left",
    )

    # Geolocation ─ median lat/lng per ZIP prefix
    geo = data["geolocation"].groupby("geolocation_zip_code_prefix").agg(
        lat=("geolocation_lat", "median"),
        lng=("geolocation_lng", "median"),
    ).reset_index()

    df = df.merge(
        geo.rename(columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "lat": "cust_lat", "lng": "cust_lng",
        }),
        on="customer_zip_code_prefix", how="left",
    )
    df = df.merge(
        geo.rename(columns={
            "geolocation_zip_code_prefix": "seller_zip_code_prefix",
            "lat": "sell_lat", "lng": "sell_lng",
        }),
        on="seller_zip_code_prefix", how="left",
    )

    # Haversine distance
    has_coords = df[["cust_lat", "sell_lat"]].notna().all(axis=1)
    df["distance_km"] = np.nan
    df.loc[has_coords, "distance_km"] = haversine_km(
        df.loc[has_coords, "cust_lat"].values,
        df.loc[has_coords, "cust_lng"].values,
        df.loc[has_coords, "sell_lat"].values,
        df.loc[has_coords, "sell_lng"].values,
    )
    print(f"   Haversine coverage: {100 * has_coords.mean():.1f}%")
    print(f"   Master dataset: {len(df):,} rows × {df.shape[1]} cols")
    return df, geo


# ── 3. Feature Engineering ───────────────────────────────────────────

def engineer_features(df):
    """
    Compute order-level features safe to derive before the train/test split.
    Seller-level aggregated features are computed in analysis.py from
    training data only, to prevent data leakage.
    """
    print("\n3  Feature engineering")
    df = df.copy()

    # Delivery timing
    df["actual_delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    df["delivery_delay_days"] = df["actual_delivery_days"] - df["estimated_delivery_days"]
    df["carrier_handling_days"] = (
        df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 86_400
    df["is_late"] = (df["delivery_delay_days"] > 0).astype(int)

    # Geographic
    df["corridor"] = df["seller_state"] + " -> " + df["customer_state"]
    df["same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)
    df["distance_band"] = pd.cut(
        df["distance_km"],
        bins=[0, 100, 500, 1_000, float("inf")],
        labels=["<100km", "100-500km", "500-1000km", ">1000km"],
    )

    # Payment type dummies
    for ptype in ["credit_card", "boleto", "debit_card"]:
        df[f"pay_{ptype}"] = (df["primary_payment_type"] == ptype).astype(int)

    # Product category dummies (top 5 by frequency)
    top_cats = df["product_category_name_english"].value_counts().head(5).index
    for cat in top_cats:
        df[f"cat_{cat.replace(' ', '_')}"] = (
            df["product_category_name_english"] == cat
        ).astype(int)

    # Temporal
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    # Binary target: satisfied = review score >= 4
    df["satisfied"] = (df["review_score"] >= 4).astype(int)

    pct = 100 * df["satisfied"].mean()
    print(f"   Satisfied: {df['satisfied'].sum():,} ({pct:.1f}%)")
    print(f"   Dissatisfied: {(~df['satisfied'].astype(bool)).sum():,} ({100 - pct:.1f}%)")
    print(f"   Enriched dataset: {len(df):,} rows × {df.shape[1]} cols")
    return df


# ── 4. Chronological Train/Test Split ────────────────────────────────

def time_split(df, train_frac=0.8):
    """80/20 chronological split to prevent temporal leakage."""
    print("\n4  Time-based split")
    
    # Safeguards chronological integrity using a direct timeframe cut-off 
    # (mimicking real production predictions) rather than random folds.
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    split_idx = int(len(df) * train_frac)

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print(f"   Train: {len(train):,}  "
          f"({train['order_purchase_timestamp'].min().date()} -> "
          f"{train['order_purchase_timestamp'].max().date()})")
    print(f"   Test:  {len(test):,}  "
          f"({test['order_purchase_timestamp'].min().date()} -> "
          f"{test['order_purchase_timestamp'].max().date()})")
    print(f"   Train satisfaction: {100 * train['satisfied'].mean():.1f}%")
    print(f"   Test  satisfaction: {100 * test['satisfied'].mean():.1f}%")

    drift = test["satisfied"].mean() - train["satisfied"].mean()
    if abs(drift) > 0.03:
        print(f"    {drift:+.1%} concept drift detected")
    return train, test


# ── Entry Point ──────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Olist data preparation pipeline")
    p.add_argument("--data-dir", default=RAW_DIR,
                   help="Path to folder containing the 9 raw Olist CSVs "
                        f"(default: {RAW_DIR})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Olist E-Commerce — Data Preparation\n")

    data = load_raw_data(args.data_dir)
    df, geo = clean_and_merge(data)
    df = engineer_features(df)
    train, test = time_split(df)

    # Export
    print("\nExporting …")
    df.to_csv(os.path.join(OUTPUT_DIR, "master_olist_data.csv"), index=False)
    train.to_csv(os.path.join(OUTPUT_DIR, "train_data.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test_data.csv"), index=False)
    geo.to_csv(os.path.join(OUTPUT_DIR, "geolocation_lookup.csv"), index=False)

    print(f"   master_olist_data.csv : {len(df):,} rows")
    print(f"   train_data.csv       : {len(train):,} rows")
    print(f"   test_data.csv        : {len(test):,} rows")
    print("\nDone.")
