"""
EcoPack Flask Application — v2
Fixes: better Gemini prompt (packaging focus), auto-fill, camera route
New:   /api/quick-score, /api/materials-for-category, /history
"""

import os, re, json, base64, uuid, traceback
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, send_from_directory)
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "ecopack-v2-secret")
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXT = {"png","jpg","jpeg","webp","gif"}
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

import sys; sys.path.insert(0, os.path.dirname(__file__))
from utils.predictor import (predict_packaging, calculate_transport_co2,
                              get_material_catalogue, get_category_rules,
                              normalise_category, CATEGORY_RULES, MATERIAL_PROPS)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY","")

GEMINI_PROMPT = """You are a packaging sustainability expert AI.
Analyse this product image and return ONLY a valid JSON object (no markdown, no extra text).

IMPORTANT: focus on what PACKAGING this product needs, NOT what the product itself is made of.
Examples:
- A bedsheet → needs cardboard box or polybag packaging (NOT glass)
- A smartphone → needs cardboard box with foam inserts
- Olive oil → comes in a glass bottle already

Return exactly this JSON:
{
  "name": "specific product name (e.g. 'Cotton Bedsheet Set', 'iPhone 15', 'Olive Oil 500ml')",
  "category": "one of: Food & Beverage / Electronics / Cosmetics / Pharmaceuticals / Toys & Games / Industrial / Agriculture / Fashion & Apparel / Household / Fresh Produce / Textile & Fabric / General",
  "material": "current packaging material (e.g. 'Cardboard', 'Plastic (PET)', 'Glass', 'Kraft Paper')",
  "fragility": "low / medium / high",
  "weight_estimate_g": estimated total packaged weight as integer,
  "description": "one sentence describing the product",
  "packaging_notes": "key packaging challenge (fragile/moisture-sensitive/heavy/bulky etc.)"
}"""

def call_gemini_vision(image_bytes: bytes, mime_type="image/jpeg") -> dict:
    """
    Call Gemini Vision using the new google-genai SDK (v1.0+).
    The old google-generativeai SDK is deprecated and broken.
    """
    if not GEMINI_API_KEY:
        return {"error": "GEMINI_API_KEY not set — add it to your environment", "fallback": True}
    try:
        import google.genai as genai
        import google.genai.types as gtypes

        client   = genai.Client(api_key=GEMINI_API_KEY)
        img_part = gtypes.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        response = client.models.generate_content(
            model    = "gemini-2.0-flash",
            contents = [GEMINI_PROMPT, img_part],
            config   = gtypes.GenerateContentConfig(
                temperature      = 0.1,   # low temp for structured extraction
                max_output_tokens= 512,
            ),
        )

        raw  = response.text.strip()
        # Strip markdown fences if Gemini wraps in ```json ... ```
        text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        data = json.loads(text)
        data["source"] = "gemini"
        return data

    except json.JSONDecodeError as e:
        # Gemini returned text but not valid JSON — return raw for debugging
        return {"error": f"JSON parse error: {e}", "raw_response": raw[:300], "fallback": True}
    except Exception as e:
        return {"error": str(e), "fallback": True}

def opencv_fallback(image_bytes: bytes) -> dict:
    try:
        import cv2, numpy as np
        nparr  = np.frombuffer(image_bytes, np.uint8)
        img    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None: return {"error":"Could not decode image"}
        h, w   = img.shape[:2]
        ar     = w / h
        pixels = img.reshape(-1,3)
        bright = int(pixels.mean())
        gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges  = cv2.Canny(gray,50,150)
        edge_d = round(float(edges.mean()),2)
        shape  = ("tall/narrow — bottle or tube" if ar < 0.7
                  else "wide/flat — sheet, box or tray" if ar > 1.5
                  else "square — jar, can or box")
        return {
            "source":"opencv","shape_hint":shape,
            "color_hint":("light/white" if bright>200 else "dark" if bright<80 else "medium"),
            "dimensions":f"{w}×{h}px","aspect_ratio":round(ar,2),
            "brightness":bright,"edge_complexity":edge_d,
            "category_hint":"container/bottle" if ar<0.8 else "box/flat/sheet",
        }
    except ImportError:
        return {"source":"opencv","error":"opencv not installed","category_hint":"unknown"}
    except Exception as e:
        return {"source":"opencv","error":str(e),"category_hint":"unknown"}

def allowed_file(fn):
    return "." in fn and fn.rsplit(".",1)[1].lower() in ALLOWED_EXT

# ══════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    history = session.get("history", [])[-5:]
    return render_template("index.html", history=history)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error":"No file uploaded"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error":"File type not allowed"}), 400

    ext   = file.filename.rsplit(".",1)[1].lower()
    fname = f"{uuid.uuid4().hex}.{ext}"
    fpath = os.path.join(app.config["UPLOAD_FOLDER"], fname)
    file.seek(0); image_bytes = file.read()
    with open(fpath,"wb") as fh: fh.write(image_bytes)

    mime_map = {"jpg":"image/jpeg","jpeg":"image/jpeg",
                "png":"image/png","webp":"image/webp","gif":"image/gif"}
    mime = mime_map.get(ext,"image/jpeg")

    gemini = call_gemini_vision(image_bytes, mime)
    opencv = opencv_fallback(image_bytes)

    session["image_file"]    = fname
    session["gemini_result"] = gemini
    session["opencv_result"] = opencv

    return jsonify({"success":True,
                    "image_url":f"/static/uploads/{fname}",
                    "gemini":gemini,"opencv":opencv})

@app.route("/confirm")
def confirm():
    gemini = session.get("gemini_result",{})
    opencv = session.get("opencv_result",{})
    image  = session.get("image_file","")
    cats   = list(CATEGORY_RULES.keys())
    mats   = list(MATERIAL_PROPS.keys())
    return render_template("confirm.html",
        gemini=gemini, opencv=opencv, image_file=image,
        materials=mats, categories=cats)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        user_data = {
            "category":         data.get("category","General"),
            "material":         data.get("material","Plastic (PET)"),
            "weight_g":         float(data.get("weight_g",500)),
            "fragility":        data.get("fragility","medium"),
            "recyclability":    float(data.get("recyclability",0.5)),
            "biodegradability": float(data.get("biodegradability",0.3)),
            "durability":       float(data.get("durability",0.7)),
        }
        result   = predict_packaging(user_data)
        dist     = float(data.get("transport_distance_km",500))
        mode     = data.get("transport_mode","road")
        wt_kg    = float(data.get("weight_g",500)) / 1000
        result["transport"] = calculate_transport_co2(wt_kg, dist, mode)

        session["prediction"]   = result
        session["product_name"] = data.get("product_name","Unknown Product")
        session["user_data"]    = user_data

        hist = session.get("history",[])
        hist.append({
            "product":     data.get("product_name","Unknown"),
            "category":    user_data["category"],
            "recommended": result["recommended_material"],
            "score":       result["recommended_sustainability"],
            "co2_saved":   result["co2_saved_kg"],
        })
        session["history"] = hist[-20:]
        return jsonify({"success":True,"result":result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error":str(e)}), 500

@app.route("/result")
def result():
    prediction   = session.get("prediction",{})
    product_name = session.get("product_name","Unknown Product")
    user_data    = session.get("user_data",{})
    image_file   = session.get("image_file","")
    gemini       = session.get("gemini_result",{})
    if not prediction: return redirect(url_for("index"))
    return render_template("result.html",
        prediction=prediction, product_name=product_name,
        user_data=user_data, image_file=image_file, gemini=gemini)

@app.route("/history")
def history():
    hist = session.get("history",[])
    return render_template("history.html", history=list(reversed(hist)))

@app.route("/compare")
def compare():
    mats = list(MATERIAL_PROPS.keys())
    cats = list(CATEGORY_RULES.keys())
    return render_template("compare.html", materials=mats, categories=cats)

# API endpoints
@app.route("/api/materials-for-category")
def api_mats_for_cat():
    cat   = request.args.get("category","General")
    rules = CATEGORY_RULES.get(normalise_category(cat), CATEGORY_RULES["General"])
    return jsonify({"allowed":rules["allowed"],"preferred":rules.get("preferred",[]),"banned":rules.get("banned",[])})

@app.route("/api/quick-score")
def api_quick_score():
    mat  = request.args.get("material","Plastic (PET)")
    p    = MATERIAL_PROPS.get(mat, MATERIAL_PROPS["Plastic (PET)"])
    co2_inv  = 1/(1+p["co2_factor"])
    cost_inv = 1/(1+p["cost_factor"])
    score = round((0.35*p["recycle"]+0.30*p["bio"]+0.20*co2_inv+0.15*cost_inv)*100,1)
    return jsonify({**p,"sustainability_score":score,"material":mat})

@app.route("/api/predict-ajax", methods=["POST"])
def api_predict_ajax():
    """Quick prediction without session — for compare tool."""
    try:
        data = request.get_json() or {}
        result = predict_packaging(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error":str(e)}), 500

@app.route("/api/materials")
def api_materials(): return jsonify(get_material_catalogue())

@app.route("/api/health")
def health():
    return jsonify({"status":"ok","gemini_configured":bool(GEMINI_API_KEY),"version":"2.0"})

@app.route("/admin")
def admin():
    try:
        df    = pd.read_csv(os.path.join(os.path.dirname(__file__),"data","packaging_dataset.csv"))
        stats = {"total_rows":len(df),"categories":df["product_category"].nunique(),
                 "materials":df["current_material"].nunique(),
                 "missing_pct":round(df.isnull().mean().mean()*100,1)}
        sample = df.sample(min(30,len(df)),random_state=1).to_dict(orient="records")
    except Exception as e:
        stats  = {"error":str(e)}
        sample = []
    return render_template("admin.html",stats=stats,sample=sample,
                           materials=get_material_catalogue())


@app.route("/api/test-gemini")
def test_gemini():
    """Test endpoint — call with a GET to verify Gemini API key works."""
    if not GEMINI_API_KEY:
        return jsonify({"ok": False, "error": "GEMINI_API_KEY env var not set"}), 400
    try:
        import google.genai as genai
        client   = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model    = "gemini-2.0-flash",
            contents = ["Reply with exactly: GEMINI_OK"],
        )
        text = response.text.strip()
        return jsonify({"ok": True, "response": text, "sdk": "google-genai",
                        "model": "gemini-2.0-flash"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/static/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    print("🌿 EcoPack AI v2")
    print(f"   Gemini: {'✅ Configured' if GEMINI_API_KEY else '⚠️  Not set (set GEMINI_API_KEY)'}")
    app.run(debug=True, host="0.0.0.0", port=5000)
