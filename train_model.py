"""
EcoPack ML Model Trainer
Cleans the dataset, engineers features, trains a GradientBoosting classifier,
and saves the model + encoders to disk.
"""

import pandas as pd
import numpy as np
import pickle
import os
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ─── Material properties lookup (for runtime scoring) ────────────────────
MATERIAL_PROPS = {
    "Plastic (PET)":       {"co2_factor": 3.8, "cost_factor": 0.4, "recycle": 0.72, "bio": 0.05, "dur": 0.80},
    "Plastic (HDPE)":      {"co2_factor": 3.5, "cost_factor": 0.35,"recycle": 0.75, "bio": 0.05, "dur": 0.85},
    "Plastic (PVC)":       {"co2_factor": 4.2, "cost_factor": 0.30,"recycle": 0.25, "bio": 0.02, "dur": 0.75},
    "Recycled Plastic":    {"co2_factor": 1.8, "cost_factor": 0.50,"recycle": 0.90, "bio": 0.10, "dur": 0.75},
    "Glass":               {"co2_factor": 1.2, "cost_factor": 0.90,"recycle": 0.95, "bio": 0.10, "dur": 0.90},
    "Aluminum":            {"co2_factor": 8.2, "cost_factor": 1.10,"recycle": 0.92, "bio": 0.02, "dur": 0.95},
    "Recycled Aluminum":   {"co2_factor": 0.8, "cost_factor": 1.20,"recycle": 0.95, "bio": 0.02, "dur": 0.92},
    "Cardboard":           {"co2_factor": 0.9, "cost_factor": 0.25,"recycle": 0.88, "bio": 0.75, "dur": 0.55},
    "Recycled Cardboard":  {"co2_factor": 0.5, "cost_factor": 0.30,"recycle": 0.92, "bio": 0.82, "dur": 0.50},
    "Kraft Paper":         {"co2_factor": 0.7, "cost_factor": 0.20,"recycle": 0.85, "bio": 0.88, "dur": 0.45},
    "Bioplastic (PLA)":    {"co2_factor": 0.8, "cost_factor": 1.40,"recycle": 0.30, "bio": 0.85, "dur": 0.65},
    "Bamboo":              {"co2_factor": 0.3, "cost_factor": 1.60,"recycle": 0.60, "bio": 0.95, "dur": 0.70},
    "Mycelium":            {"co2_factor": 0.1, "cost_factor": 2.00,"recycle": 0.40, "bio": 0.99, "dur": 0.50},
    "Hemp Fiber":          {"co2_factor": 0.2, "cost_factor": 1.70,"recycle": 0.65, "bio": 0.96, "dur": 0.60},
    "Steel (Tinplate)":    {"co2_factor": 3.0, "cost_factor": 0.70,"recycle": 0.88, "bio": 0.02, "dur": 0.97},
    "Wood":                {"co2_factor": 0.5, "cost_factor": 1.30,"recycle": 0.70, "bio": 0.80, "dur": 0.85},
    "Styrofoam (EPS)":     {"co2_factor": 3.3, "cost_factor": 0.20,"recycle": 0.10, "bio": 0.01, "dur": 0.60},
    "Seaweed Film":        {"co2_factor": 0.05,"cost_factor": 2.20,"recycle": 0.45, "bio": 0.98, "dur": 0.35},
}

# ─── Text normalisation helpers ──────────────────────────────────────────
MATERIAL_ALIASES = {
    r"pet|polyethylene\s*terephthalate|pet\s*bottle|pet\s*plastic": "Plastic (PET)",
    r"hdpe|high\s*density\s*pe|hdpe\s*container|hdpe\s*plastic": "Plastic (HDPE)",
    r"pvc|polyvinyl": "Plastic (PVC)",
    r"recycled\s*plastic|r-pet|rplastic": "Recycled Plastic",
    r"glass|borosilicate|soda.lime": "Glass",
    r"alumin[ui]m|al\s*can|alum$": "Aluminum",
    r"recycled\s*alum": "Recycled Aluminum",
    r"cardboard|corrugated|paperboard|carton\s*board|card\s*board": "Cardboard",
    r"recycled\s*cardboard|r-cardboard": "Recycled Cardboard",
    r"kraft|brown\s*paper|unbleached\s*kraft": "Kraft Paper",
    r"pla|polylactic|bioplastic|bio.plastic": "Bioplastic (PLA)",
    r"bamboo": "Bamboo",
    r"mycelium|mushroom\s*pack": "Mycelium",
    r"hemp": "Hemp Fiber",
    r"steel|tinplate|tin": "Steel (Tinplate)",
    r"wood|wooden": "Wood",
    r"styrofoam|eps|expanded\s*polystyrene": "Styrofoam (EPS)",
    r"seaweed": "Seaweed Film",
}

CATEGORY_MAP = {
    r"food|beverage|drink|snack|cereal": "Food & Beverage",
    r"electronic|tech|gadget|device|laptop|phone|computer": "Electronics",
    r"cosmetic|beauty|makeup|perfume|skincare": "Cosmetics",
    r"pharma|medicine|medical|health|drug": "Pharmaceuticals",
    r"toy|game|play|puzzle": "Toys & Games",
    r"industri|tool|machine|chemical|industrial": "Industrial",
    r"agri|farm|seed|garden|plant": "Agriculture",
    r"fashion|apparel|cloth|shoe|jewel": "Fashion & Apparel",
    r"household|home|kitchen|clean|detergent": "Household",
    r"produce|fruit|vegetable|fresh": "Fresh Produce",
}

def normalise_material(text):
    if pd.isna(text):
        return "Unknown"
    t = str(text).lower().strip()
    for pattern, canonical in MATERIAL_ALIASES.items():
        if re.search(pattern, t):
            return canonical
    return "Plastic (PET)"  # default fallback

def normalise_category(text):
    if pd.isna(text):
        return "General"
    t = str(text).lower().strip()
    for pattern, canonical in CATEGORY_MAP.items():
        if re.search(pattern, t):
            return canonical
    return "General"

def parse_weight(val):
    """Convert weight to grams, handling mixed units."""
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    try:
        if "kg" in s:
            return float(re.findall(r"[\d.]+", s)[0]) * 1000
        return float(re.findall(r"[\d.]+", s)[0])
    except:
        return np.nan

def fragility_to_num(val):
    mapping = {"low": 1, "medium": 2, "high": 3}
    if isinstance(val, (int, float)):
        return val
    if pd.isna(val):
        return np.nan
    return mapping.get(str(val).lower().strip(), np.nan)

# ─── Load and clean data ─────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "packaging_dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Raw rows: {len(df)} | Columns: {len(df.columns)}")

# Drop duplicates
df.drop_duplicates(inplace=True)
print(f"   After dedup: {len(df)}")

# Normalise text
df["current_material_clean"] = df["current_material"].apply(normalise_material)
df["category_clean"] = df["product_category"].apply(normalise_category)

# Parse weight
df["weight_g"] = df["weight"].apply(parse_weight)

# Fragility
df["fragility_num"] = df["fragility"].apply(fragility_to_num)

# Fill missing numerics with median
for col in ["weight_g", "fragility_num", "co2_emission_kg", "cost_per_unit", 
            "recyclability", "biodegradability", "durability"]:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# Remove extreme outliers (>99th percentile for co2 and cost)
df = df[df["co2_emission_kg"] < df["co2_emission_kg"].quantile(0.99)]
df = df[df["cost_per_unit"]   < df["cost_per_unit"].quantile(0.99)]
print(f"   After outlier removal: {len(df)}")

# ─── Sustainability score ─────────────────────────────────────────────────
def compute_sustainability_score(row):
    co2_inv  = 1 / (1 + row["co2_emission_kg"])
    cost_inv = 1 / (1 + row["cost_per_unit"])
    score = (
        0.35 * row["recyclability"] +
        0.30 * row["biodegradability"] +
        0.20 * co2_inv +
        0.15 * cost_inv
    )
    return round(min(score, 1.0), 4)

df["sustainability_score"] = df.apply(compute_sustainability_score, axis=1)

# ─── Feature engineering ──────────────────────────────────────────────────
CATEGORIES = ["Food & Beverage","Electronics","Cosmetics","Pharmaceuticals",
              "Toys & Games","Industrial","Agriculture","Fashion & Apparel",
              "Household","Fresh Produce","General"]

cat_encoder = LabelEncoder()
cat_encoder.classes_ = np.array(CATEGORIES)

mat_encoder = LabelEncoder()
mat_encoder.fit(df["current_material_clean"])

df["category_enc"] = cat_encoder.transform(
    df["category_clean"].apply(lambda x: x if x in CATEGORIES else "General")
)
df["material_enc"] = mat_encoder.transform(df["current_material_clean"])

FEATURE_COLS = ["category_enc", "material_enc", "weight_g", "fragility_num",
                "co2_emission_kg", "cost_per_unit", "recyclability",
                "biodegradability", "durability", "sustainability_score"]

TARGET_COL = "recommended_material"

# Clean target
df = df[df[TARGET_COL].notna()]
target_encoder = LabelEncoder()
df["target_enc"] = target_encoder.fit_transform(df[TARGET_COL])

X = df[FEATURE_COLS].values
y = df["target_enc"].values

# Final NaN imputation on features
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median")
X = imp.fit_transform(X)

# ─── Train/test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─── Model training ───────────────────────────────────────────────────────
print("\n🤖 Training GradientBoosting model...")
model = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    min_samples_split=5, random_state=42
)
model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print(f"   ✅ Test Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_, zero_division=0))

cv_scores = cross_val_score(model, X_train_s, y_train, cv=5)
print(f"   Cross-val mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── Save artefacts ───────────────────────────────────────────────────────
artifacts = {
    "model":          model,
    "scaler":         scaler,
    "imputer":        imp,
    "cat_encoder":    cat_encoder,
    "mat_encoder":    mat_encoder,
    "target_encoder": target_encoder,
    "feature_cols":   FEATURE_COLS,
    "material_props": MATERIAL_PROPS,
    "categories":     CATEGORIES,
    "category_map":   CATEGORY_MAP,
    "material_aliases": MATERIAL_ALIASES,
    "accuracy":       acc,
}

pkl_path = os.path.join(MODEL_DIR, "ecopack_model.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(artifacts, f)

print(f"\n💾 Model saved → {pkl_path}")
print("🎉 Training complete!")
