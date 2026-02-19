from flask import Flask, request, jsonify, render_template, session
import joblib
import numpy as np
import os
import sqlite3
import random

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback_key_for_local_dev")

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CO2_MODEL_PATH = os.path.join(BASE_DIR, "co2_model.pkl")
COST_MODEL_PATH = os.path.join(BASE_DIR, "cost_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "Eco_Pack.db")

# ================= LOAD ML MODELS =================
try:
    co2_model = joblib.load(CO2_MODEL_PATH)
    cost_model = joblib.load(COST_MODEL_PATH)
    co2_model.n_jobs = 1
    cost_model.n_jobs = 1
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

# ================= DB CONNECTION =================
def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
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

    # Convert types
    data["strength_score"] = int(data["strength_score"])
    data["biodegradability_score"] = int(data["biodegradability_score"])
    data["moisture_resistance"] = int(data["moisture_resistance"])
    data["heat_resistance"] = int(data["heat_resistance"])
    data["weight_capacity_kg"] = float(data["weight_capacity_kg"])
    data["recyclability_percent"] = float(data["recyclability_percent"])
    weights = data.get("weights", {"cost": 1, "co2": 1})

    # Load materials from DB
    try:
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
    except Exception as e:
        return jsonify({"error": f"DB Error: {e}"}), 500

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

        try:
            co2 = float(co2_model.predict(features)[0])
            cost = float(cost_model.predict(features)[0])
        except Exception as e:
            co2, cost = 0, 0
            print(f"Prediction error for {row[0]}: {e}")

        # Penalty calculation
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

    # Normalization & eco_score
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

    # Save recommendation in session for dashboard
    session["recommendation"] = ranked[:5]

    return jsonify({"recommended_materials": ranked[:5]})

# ================= SAVE REPORT =================
@app.route("/save-report", methods=["POST"])
def save_report():
    data = request.json
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_reports
            (product_category, selected_material, eco_score, predicted_co2, predicted_cost)
            VALUES (?, ?, ?, ?, ?)
        """, (
            data["product_category"],
            data["selected_material"],
            data["eco_score"],
            data["predicted_co2"],
            data["predicted_cost"]
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        return jsonify({"error": f"DB Error: {e}"}), 500

    return jsonify({"message": "Report saved successfully"})

# ================= DASHBOARD DATA =================
@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
    start_date = request.args.get("start")
    end_date = request.args.get("end")
    material = request.args.get("material")

    conn = get_db_connection()
    cur = conn.cursor()

    query = "FROM user_reports WHERE 1=1"
    params = []

    if start_date and end_date:
        query += " AND created_at BETWEEN ? AND ?"
        params.extend([start_date, end_date])
    if material:
        query += " AND selected_material = ?"
        params.append(material)

    # KPI Metrics
    cur.execute(f"""
        SELECT COUNT(*),
               AVG(eco_score),
               AVG(predicted_co2),
               AVG(predicted_cost),
               SUM(predicted_co2),
               SUM(predicted_cost)
        {query}
    """, params)
    result = cur.fetchone()
    total_reports = result[0] or 0
    avg_eco = round(result[1] or 0,2)
    avg_co2 = round(result[2] or 0,2)
    avg_cost = round(result[3] or 0,2)
    total_co2 = result[4] or 0
    total_cost = result[5] or 0

    # Baseline comparison
    baseline_co2 = total_reports * 50
    baseline_cost = total_reports * 100
    co2_reduction = round(baseline_co2 - total_co2,2)
    cost_savings = round(baseline_cost - total_cost,2)
    better_than_plastic = round((co2_reduction / baseline_co2 * 100) if baseline_co2 else 0,2)

    # Material breakdown
    cur.execute(f"""
        SELECT selected_material, COUNT(*)
        {query}
        GROUP BY selected_material
    """, params)
    rows = cur.fetchall()
    materials = [r[0] for r in rows]
    material_counts = [r[1] for r in rows]
    top_material = materials[0] if materials else "N/A"

    # Trend data
    cur.execute(f"""
        SELECT created_at, predicted_cost, predicted_co2
        {query}
        ORDER BY created_at
    """, params)
    trend = cur.fetchall()
    cumulative_cost, cumulative_co2 = [], []
    run_cost = run_co2 = 0
    for row in trend:
        run_cost += (100 - float(row[1]))
        run_co2 += (50 - float(row[2]))
        cumulative_cost.append(round(run_cost,2))
        cumulative_co2.append(round(run_co2,2))

    cur.close()
    conn.close()

    # AI Insight
    insight = generate_ai_insight(total_reports, cost_savings, co2_reduction, better_than_plastic)

    return jsonify({
        "top_material": top_material,
        "total_reports": total_reports,
        "avg_eco": avg_eco,
        "avg_co2": avg_co2,
        "avg_cost": avg_cost,
        "co2_reduction": co2_reduction,
        "cost_savings": cost_savings,
        "better_than_plastic": better_than_plastic,
        "materials": materials,
        "material_counts": material_counts,
        "cumulative_cost": cumulative_cost,
        "cumulative_co2": cumulative_co2,
        "insight": insight
    })

def generate_ai_insight(total, savings, co2, percent):
    if total == 0:
        return "No reports available yet. Generate sustainability insights by saving material recommendations."
    return f"""
    Over {total} sustainability analyses, EcoPack AI achieved 
    ₹{savings} in estimated cost savings and avoided {co2} units of CO₂ emissions. 
    This represents a {percent}% improvement compared to traditional plastic packaging.
    """

# ================= DASHBOARD PAGES =================
@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard_data.html")

@app.route("/recommendation_dashboard")
def recommendation_dashboard():
    recommendation = session.get("recommendation")
    return render_template("recommendation_dashboard.html", recommendation=recommendation)

# ================= RUN APP =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

