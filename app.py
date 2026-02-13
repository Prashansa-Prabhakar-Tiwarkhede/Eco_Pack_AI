from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

import sqlite3

import random

app = Flask(__name__)

# Load ML models
co2_model = joblib.load("co2_model.pkl")
cost_model = joblib.load("cost_model.pkl")

co2_model.n_jobs = 1
cost_model.n_jobs = 1

# ================= DB CONNECTION =================
def get_db_connection():
    db_path = os.path.join(os.getcwd(), "Eco_Pack.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn



# ================= LEVEL CONVERSION =================
def level_to_num(val):
    if isinstance(val, str):
        val = val.strip().capitalize()
    mapping = {"Low": 3, "Medium": 6, "High": 9}
    return mapping.get(val, 0)

# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    required_fields = [
        "strength_score",
        "weight_capacity_kg",
        "biodegradability_score",
        "recyclability_percent",
        "moisture_resistance",
        "heat_resistance"
    ]

    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing field: {field}"}), 400

    data["strength_score"] = int(data["strength_score"])
    data["biodegradability_score"] = int(data["biodegradability_score"])
    data["moisture_resistance"] = int(data["moisture_resistance"])
    data["heat_resistance"] = int(data["heat_resistance"])
    data["weight_capacity_kg"] = float(data["weight_capacity_kg"])
    data["recyclability_percent"] = float(data["recyclability_percent"])

    weights = data.get("weights", {"cost": 1, "co2": 1})

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT material_name,
               strength_score,
               CAST(weight_capacity_kg AS REAL),
               biodegradability_score,
               CAST(recyclability_percent AS REAL),
               moisture_resistance,
               heat_resistance
        FROM materials
    """)

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return jsonify({"error": "No materials in database"}), 500

    results = []

    for row in rows:
        strength = level_to_num(row[1])
        weight = float(row[2])
        biodeg = level_to_num(row[3])
        recycle = float(row[4])
        moisture = level_to_num(row[5])
        heat = level_to_num(row[6])

        features = np.array([[strength, weight, biodeg, recycle, moisture, heat]])

        co2 = float(co2_model.predict(features)[0])
        cost = float(cost_model.predict(features)[0])

        # Penalty calculation (ranking instead of rejection)
        penalty = 0

        if strength < data["strength_score"]:
            penalty += data["strength_score"] - strength

        if weight < data["weight_capacity_kg"]:
            penalty += data["weight_capacity_kg"] - weight

        if biodeg < data["biodegradability_score"]:
            penalty += data["biodegradability_score"] - biodeg

        if recycle < data["recyclability_percent"]:
            penalty += data["recyclability_percent"] - recycle

        if moisture < data["moisture_resistance"]:
            penalty += data["moisture_resistance"] - moisture

        if heat < data["heat_resistance"]:
            penalty += data["heat_resistance"] - heat

        results.append({
            "material": row[0],
            "predicted_co2": round(co2, 2),
            "predicted_cost": round(cost, 2),
            "penalty": penalty,
            "confidence": random.randint(80, 95)
        })

    max_cost = max(r["predicted_cost"] for r in results) or 1
    max_co2 = max(r["predicted_co2"] for r in results) or 1

    for r in results:
        cost_norm = r["predicted_cost"] / max_cost
        co2_norm = r["predicted_co2"] / max_co2

        r["eco_score"] = round(
            weights.get("cost", 1) * cost_norm +
            weights.get("co2", 1) * co2_norm +
            r["penalty"] * 0.1,
            3
        )

    ranked = sorted(results, key=lambda x: x["eco_score"])

    return jsonify({
        "recommended_materials": ranked[:5],
        "rejected_materials": rejected
    })



# ================= SAVE REPORT =================
@app.route("/save-report", methods=["POST"])
def save_report():
    data = request.get_json()

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO user_reports
        (product_category, selected_material, eco_score, predicted_co2, predicted_cost)
        VALUES (?,?,?,?,?)
    """, (

        data["product_category"],
        data["material"],
        data["eco_score"],
        data["predicted_co2"],
        data["predicted_cost"]
    ))

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": "Report saved successfully"})

@app.route("/materials", methods=["GET"])
def get_materials():
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT material_name,
                   strength_score,
                   weight_capacity_kg,
                   biodegradability_score,
                   recyclability_percent,
                   moisture_resistance,
                   heat_resistance
            FROM materials
        """)

        rows = cur.fetchall()
        cur.close()
        conn.close()

        materials = []

        for row in rows:
            features = np.array([[ 
                level_to_num(row[1]),
                float(row[2]),
                level_to_num(row[3]),
                float(row[4]),
                level_to_num(row[5]),
                level_to_num(row[6])
            ]])

            co2 = float(co2_model.predict(features)[0])
            cost = float(cost_model.predict(features)[0])

            materials.append({
                "material": row[0],
                "predicted_co2": round(co2, 2),
                "predicted_cost": round(cost, 2)
            })

        # Normalization (same as /predict)
        max_cost = max(m["predicted_cost"] for m in materials) or 1
        max_co2 = max(m["predicted_co2"] for m in materials) or 1

        for m in materials:
            cost_norm = m["predicted_cost"] / max_cost
            co2_norm = m["predicted_co2"] / max_co2
            m["eco_score"] = round((cost_norm + co2_norm) / 2, 3)

        return jsonify(materials)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)









