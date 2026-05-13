"""
EcoPack Predictor — v2
Fixes:
  1. Hard rule-based constraint layer (bedsheet → never Glass)
  2. Category-material compatibility matrix
  3. Sustainability scoring improvements
  4. Packaging purpose detection (not product material)
"""

import pickle, os, re, numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ecopack_model.pkl")
_artifacts = None

def load_model():
    global _artifacts
    if _artifacts is None:
        with open(MODEL_PATH, "rb") as f:
            _artifacts = pickle.load(f)
    return _artifacts

CATEGORY_RULES = {
    "Food & Beverage": {
        "allowed":   ["Glass","Recycled Aluminum","Aluminum","Bioplastic (PLA)",
                      "Recycled Plastic","Plastic (PET)","Recycled Cardboard",
                      "Kraft Paper","Steel (Tinplate)","Seaweed Film","Hemp Fiber"],
        "banned":    ["Mycelium","Styrofoam (EPS)","Wood","Bamboo"],
        "preferred": ["Glass","Bioplastic (PLA)","Recycled Aluminum","Seaweed Film"],
    },
    "Electronics": {
        "allowed":   ["Recycled Cardboard","Cardboard","Mycelium","Hemp Fiber",
                      "Bamboo","Recycled Plastic","Wood"],
        "banned":    ["Glass","Seaweed Film","Styrofoam (EPS)","Aluminum"],
        "preferred": ["Recycled Cardboard","Mycelium","Hemp Fiber","Bamboo"],
    },
    "Cosmetics": {
        "allowed":   ["Glass","Recycled Aluminum","Aluminum","Bamboo",
                      "Bioplastic (PLA)","Recycled Plastic","Recycled Cardboard"],
        "banned":    ["Styrofoam (EPS)","Mycelium","Seaweed Film","Steel (Tinplate)"],
        "preferred": ["Glass","Recycled Aluminum","Bamboo","Bioplastic (PLA)"],
    },
    "Pharmaceuticals": {
        "allowed":   ["Glass","Recycled Cardboard","Bioplastic (PLA)",
                      "Plastic (HDPE)","Recycled Plastic","Aluminum"],
        "banned":    ["Mycelium","Seaweed Film","Bamboo","Wood","Hemp Fiber"],
        "preferred": ["Glass","Recycled Cardboard","Bioplastic (PLA)"],
    },
    "Toys & Games": {
        "allowed":   ["Recycled Cardboard","Cardboard","Bamboo",
                      "Recycled Plastic","Wood","Hemp Fiber"],
        "banned":    ["Glass","Styrofoam (EPS)","Seaweed Film","Aluminum"],
        "preferred": ["Recycled Cardboard","Bamboo","Wood"],
    },
    "Industrial": {
        "allowed":   ["Recycled Aluminum","Aluminum","Steel (Tinplate)",
                      "Recycled Cardboard","Wood","Plastic (HDPE)","Recycled Plastic"],
        "banned":    ["Glass","Seaweed Film","Mycelium","Bioplastic (PLA)"],
        "preferred": ["Recycled Aluminum","Steel (Tinplate)","Recycled Cardboard"],
    },
    "Agriculture": {
        "allowed":   ["Kraft Paper","Hemp Fiber","Bamboo","Bioplastic (PLA)",
                      "Recycled Cardboard","Recycled Plastic"],
        "banned":    ["Glass","Aluminum","Steel (Tinplate)","Styrofoam (EPS)"],
        "preferred": ["Kraft Paper","Hemp Fiber","Bioplastic (PLA)"],
    },
    "Fashion & Apparel": {
        "allowed":   ["Recycled Cardboard","Kraft Paper","Bamboo",
                      "Recycled Plastic","Hemp Fiber","Bioplastic (PLA)"],
        "banned":    ["Glass","Aluminum","Steel (Tinplate)","Mycelium",
                      "Seaweed Film","Styrofoam (EPS)"],
        "preferred": ["Recycled Cardboard","Kraft Paper","Bamboo","Hemp Fiber"],
    },
    "Household": {
        "allowed":   ["Recycled Cardboard","Recycled Plastic","Glass",
                      "Bioplastic (PLA)","Bamboo","Kraft Paper","Recycled Aluminum"],
        "banned":    ["Styrofoam (EPS)","Seaweed Film","Mycelium"],
        "preferred": ["Recycled Cardboard","Bioplastic (PLA)","Recycled Plastic"],
    },
    "Fresh Produce": {
        "allowed":   ["Mycelium","Seaweed Film","Kraft Paper","Bioplastic (PLA)",
                      "Recycled Cardboard","Hemp Fiber","Bamboo"],
        "banned":    ["Glass","Aluminum","Steel (Tinplate)","Styrofoam (EPS)",
                      "Plastic (PVC)","Plastic (HDPE)","Plastic (PET)"],
        "preferred": ["Mycelium","Seaweed Film","Bioplastic (PLA)","Kraft Paper"],
    },
    "Textile & Fabric": {
        "allowed":   ["Recycled Cardboard","Kraft Paper","Recycled Plastic",
                      "Hemp Fiber","Bamboo","Bioplastic (PLA)"],
        "banned":    ["Glass","Aluminum","Steel (Tinplate)","Mycelium",
                      "Seaweed Film","Styrofoam (EPS)","Wood"],
        "preferred": ["Recycled Cardboard","Kraft Paper","Hemp Fiber"],
    },
    "General": {
        "allowed":   ["Recycled Cardboard","Kraft Paper","Bamboo","Hemp Fiber",
                      "Bioplastic (PLA)","Recycled Plastic","Recycled Aluminum","Glass"],
        "banned":    ["Styrofoam (EPS)","Plastic (PVC)"],
        "preferred": ["Recycled Cardboard","Kraft Paper","Bamboo"],
    },
}

MATERIAL_PROPS = {
    "Plastic (PET)":      {"co2_factor":3.8,"cost_factor":0.40,"recycle":0.72,"bio":0.05,"dur":0.80},
    "Plastic (HDPE)":     {"co2_factor":3.5,"cost_factor":0.35,"recycle":0.75,"bio":0.05,"dur":0.85},
    "Plastic (PVC)":      {"co2_factor":4.2,"cost_factor":0.30,"recycle":0.25,"bio":0.02,"dur":0.75},
    "Recycled Plastic":   {"co2_factor":1.8,"cost_factor":0.50,"recycle":0.90,"bio":0.10,"dur":0.75},
    "Glass":              {"co2_factor":1.2,"cost_factor":0.90,"recycle":0.95,"bio":0.10,"dur":0.90},
    "Aluminum":           {"co2_factor":8.2,"cost_factor":1.10,"recycle":0.92,"bio":0.02,"dur":0.95},
    "Recycled Aluminum":  {"co2_factor":0.8,"cost_factor":1.20,"recycle":0.95,"bio":0.02,"dur":0.92},
    "Cardboard":          {"co2_factor":0.9,"cost_factor":0.25,"recycle":0.88,"bio":0.75,"dur":0.55},
    "Recycled Cardboard": {"co2_factor":0.5,"cost_factor":0.30,"recycle":0.92,"bio":0.82,"dur":0.50},
    "Kraft Paper":        {"co2_factor":0.7,"cost_factor":0.20,"recycle":0.85,"bio":0.88,"dur":0.45},
    "Bioplastic (PLA)":   {"co2_factor":0.8,"cost_factor":1.40,"recycle":0.30,"bio":0.85,"dur":0.65},
    "Bamboo":             {"co2_factor":0.3,"cost_factor":1.60,"recycle":0.60,"bio":0.95,"dur":0.70},
    "Mycelium":           {"co2_factor":0.1,"cost_factor":2.00,"recycle":0.40,"bio":0.99,"dur":0.50},
    "Hemp Fiber":         {"co2_factor":0.2,"cost_factor":1.70,"recycle":0.65,"bio":0.96,"dur":0.60},
    "Steel (Tinplate)":   {"co2_factor":3.0,"cost_factor":0.70,"recycle":0.88,"bio":0.02,"dur":0.97},
    "Wood":               {"co2_factor":0.5,"cost_factor":1.30,"recycle":0.70,"bio":0.80,"dur":0.85},
    "Styrofoam (EPS)":    {"co2_factor":3.3,"cost_factor":0.20,"recycle":0.10,"bio":0.01,"dur":0.60},
    "Seaweed Film":       {"co2_factor":0.05,"cost_factor":2.20,"recycle":0.45,"bio":0.98,"dur":0.35},
}

CATEGORY_MAP = {
    r"food|beverage|drink|snack|cereal|juice|dairy|meat|bakery|spice|sauce|coffee|tea|alcohol|wine|beer": "Food & Beverage",
    r"electronic|tech|gadget|device|laptop|phone|tablet|computer|camera|headphone|speaker|charger|battery": "Electronics",
    r"cosmetic|beauty|makeup|perfume|skincare|serum|lotion|cream|lipstick|mascara|shampoo|conditioner|deodorant": "Cosmetics",
    r"pharma|medicine|medical|health|drug|vitamin|supplement|capsule|syrup|ointment|bandage|syringe": "Pharmaceuticals",
    r"toy|game|play|puzzle|lego|doll|board game|action figure|card game": "Toys & Games",
    r"industri|tool|machine|chemical|lubricant|paint|adhesive|hardware|bolt|screw": "Industrial",
    r"agri|farm|seed|garden|plant|fertilizer|pesticide|soil|compost": "Agriculture",
    r"fashion|apparel|cloth|shoe|jewel|watch|handbag|wallet|belt|tie|dress|shirt|pant|jeans|sock": "Fashion & Apparel",
    r"household|home|kitchen|clean|detergent|candle|vase|cutlery|plate|bowl": "Household",
    r"produce|fruit|vegetable|fresh|organic|salad|herb|mushroom|berry|apple|banana": "Fresh Produce",
    r"textile|fabric|bedsheet|bed sheet|blanket|pillow|towel|curtain|linen|quilt|duvet|mattress|rug|carpet": "Textile & Fabric",
}

MATERIAL_ALIASES = {
    r"pet|polyethylene\s*terephthalate|pet\s*bottle|pet\s*plastic": "Plastic (PET)",
    r"hdpe|high\s*density\s*pe": "Plastic (HDPE)",
    r"pvc|polyvinyl": "Plastic (PVC)",
    r"recycled\s*plastic|r-pet": "Recycled Plastic",
    r"glass|borosilicate|soda.lime": "Glass",
    r"alumin[ui]m|al\s*can|alum$": "Aluminum",
    r"recycled\s*alum": "Recycled Aluminum",
    r"cardboard|corrugated|paperboard|carton": "Cardboard",
    r"recycled\s*cardboard": "Recycled Cardboard",
    r"kraft|brown\s*paper": "Kraft Paper",
    r"pla|polylactic|bioplastic|bio.plastic": "Bioplastic (PLA)",
    r"bamboo": "Bamboo",
    r"mycelium|mushroom\s*pack": "Mycelium",
    r"hemp": "Hemp Fiber",
    r"steel|tinplate|tin": "Steel (Tinplate)",
    r"wood|wooden": "Wood",
    r"styrofoam|eps|expanded\s*polystyrene|foam": "Styrofoam (EPS)",
    r"seaweed": "Seaweed Film",
}

def normalise_category(text: str) -> str:
    t = str(text).lower().strip()
    for pattern, canonical in CATEGORY_MAP.items():
        if re.search(pattern, t):
            return canonical
    return "General"

def normalise_material(text: str) -> str:
    t = str(text).lower().strip()
    for pattern, canonical in MATERIAL_ALIASES.items():
        if re.search(pattern, t):
            return canonical
    return "Plastic (PET)"

FRAGILITY_MAP = {"low": 1, "medium": 2, "high": 3}
def fragility_num(val):
    if isinstance(val, (int, float)): return float(val)
    return float(FRAGILITY_MAP.get(str(val).lower().strip(), 2))

def _compute_score(props: dict) -> float:
    if not props: return 0.0
    co2_inv  = 1 / (1 + props.get("co2_factor", 3))
    cost_inv = 1 / (1 + props.get("cost_factor", 0.5))
    return round((0.35*props.get("recycle",0.5) +
                  0.30*props.get("bio",0.1) +
                  0.20*co2_inv + 0.15*cost_inv) * 100, 1)

def predict_packaging(user_data: dict) -> dict:
    arts       = load_model()
    model      = arts["model"]
    scaler     = arts["scaler"]
    imputer    = arts["imputer"]
    target_enc = arts["target_encoder"]
    mat_enc    = arts["mat_encoder"]

    category = normalise_category(user_data.get("category", "General"))
    material = normalise_material(user_data.get("material", "Plastic (PET)"))
    weight_g = float(user_data.get("weight_g", 500))
    frag     = fragility_num(user_data.get("fragility", "medium"))

    props    = MATERIAL_PROPS.get(material, MATERIAL_PROPS["Plastic (PET)"])
    co2      = float(user_data.get("co2_per_kg",   props["co2_factor"] * weight_g / 1000))
    cost     = float(user_data.get("cost_per_unit", props["cost_factor"] * weight_g / 500))
    rec      = float(user_data.get("recyclability",    props["recycle"]))
    bio      = float(user_data.get("biodegradability", props["bio"]))
    dur      = float(user_data.get("durability",       props["dur"]))
    sus_score = 0.35*rec + 0.30*bio + 0.20/(1+co2) + 0.15/(1+cost)

    CATEGORIES = list(arts["categories"]) + ["Textile & Fabric","General"]
    cat_idx    = CATEGORIES.index(category) if category in CATEGORIES else len(CATEGORIES)-1
    mat_classes = list(mat_enc.classes_)
    mat_idx    = mat_classes.index(material) if material in mat_classes else 0

    features   = np.array([[cat_idx, mat_idx, weight_g, frag,
                             co2, cost, rec, bio, dur, sus_score]])
    feat_s     = scaler.transform(imputer.transform(features))
    raw_proba  = model.predict_proba(feat_s)[0]
    all_mats   = list(target_enc.classes_)

    rules    = CATEGORY_RULES.get(category, CATEGORY_RULES["General"])
    allowed  = set(rules["allowed"])
    banned   = set(rules["banned"])
    adjusted = raw_proba.copy()

    for i, mat in enumerate(all_mats):
        if mat in banned:        adjusted[i] = 0.0
        elif mat not in allowed: adjusted[i] *= 0.05

    for mat in rules.get("preferred", []):
        if mat in all_mats:
            adjusted[all_mats.index(mat)] *= 2.5

    total = adjusted.sum()
    if total > 0:
        adjusted /= total
    else:
        for mat in rules["preferred"]:
            if mat in all_mats:
                adjusted[all_mats.index(mat)] = 1.0 / len(rules["preferred"])
        adjusted /= adjusted.sum()

    pred_idx    = int(adjusted.argmax())
    recommended = all_mats[pred_idx]
    confidence  = float(adjusted[pred_idx])
    top3_idx    = adjusted.argsort()[-3:][::-1]
    top3        = [{"material": all_mats[i], "confidence": round(float(adjusted[i]),4)}
                   for i in top3_idx]

    rec_props  = MATERIAL_PROPS.get(recommended, {})
    curr_props = MATERIAL_PROPS.get(material, {})
    wt_kg      = weight_g / 1000
    curr_co2   = round(curr_props.get("co2_factor", 3.8) * wt_kg, 5)
    rec_co2    = round(rec_props.get("co2_factor",  0.5) * wt_kg, 5)
    co2_saved  = round(max(0, curr_co2 - rec_co2), 5)

    return {
        "recommended_material":       recommended,
        "confidence":                 round(confidence, 4),
        "top3_recommendations":       top3,
        "current_material":           material,
        "current_co2":                curr_co2,
        "recommended_co2":            rec_co2,
        "co2_saved_kg":               co2_saved,
        "co2_reduction_pct":          round(co2_saved / (curr_co2 + 1e-9) * 100, 1),
        "current_sustainability":     _compute_score(curr_props),
        "recommended_sustainability": _compute_score(rec_props),
        "sustainability_improvement": round(_compute_score(rec_props) - _compute_score(curr_props), 1),
        "current_props":              curr_props,
        "recommended_props":          rec_props,
        "category_used":              category,
        "engineered_features": {
            "category": category, "material": material, "weight_g": weight_g,
            "fragility": frag, "co2": co2, "cost": cost,
            "recyclability": rec, "biodegradability": bio,
            "durability": dur, "sustainability_score": round(sus_score*100, 1),
        },
    }

TRANSPORT_FACTORS = {"road": 0.096, "rail": 0.028, "sea": 0.010, "air": 0.602}

def calculate_transport_co2(weight_kg, distance_km, mode="road"):
    factor = TRANSPORT_FACTORS.get(mode, 0.096)
    co2_kg = round(factor * (weight_kg / 1000) * distance_km, 6)
    return {"weight_kg": weight_kg, "distance_km": distance_km,
            "mode": mode, "factor": factor, "transport_co2_kg": co2_kg}

def get_material_catalogue():
    return MATERIAL_PROPS

def get_category_rules():
    return CATEGORY_RULES
