# 🌿 EcoPack AI — Sustainable Packaging Intelligence System

A full-stack AI application that analyses product images and recommends the most
sustainable packaging material using Gemini Vision API, OpenCV, and a trained
Gradient Boosting ML model.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repo>
cd ecopack
pip install -r requirements.txt
```

### 2. Configure Gemini API Key

```bash
export GEMINI_API_KEY="your-key-here"
# Windows: set GEMINI_API_KEY=your-key-here
```

Get a free key at: https://makersuite.google.com/app/apikey

### 3. Train Model

```bash
python train_model.py              # trains model → models/ecopack_model.pkl
```

### 4. Run the App

```bash
python app.py
# Open http://localhost:5000
```

---

## 📁 Project Structure

```
ecopack/
├── app.py                    ← Flask application (routes, Gemini, OpenCV)
├── train_model.py            ← ML training script
├── requirements.txt
├── data/
│   └── packaging_dataset.csv ← Dataset (1470 rows)
├── models/
│   └── ecopack_model.pkl     ← Trained GradientBoosting + encoders
├── utils/
│   └── predictor.py          ← Feature engineering + prediction engine
├── templates/
│   ├── index.html            ← Upload page
│   ├── confirm.html          ← AI analysis review + editing
│   ├── result.html           ← Full results with charts
│   └── admin.html            ← Admin panel
└── static/
    ├── css/style.css         ← Full dark eco theme stylesheet
    └── uploads/              ← Uploaded images (auto-cleared)
```

---

## 🧠 How It Works

### Step 1 — Upload Image
- User uploads any product image
- System sends it to **Gemini Vision API** with a structured prompt
- Returns: product name, category, material, fragility, description
- **OpenCV fallback**: shape detection, colour analysis, dimensions

### Step 2 — Confirm & Edit
- User reviews AI-detected attributes
- Can override: name, category, material, fragility
- Sets packaging requirements: weight, strength, moisture/heat resistance
- Sets transport details for CO₂ calculation

### Step 3 — ML Recommendation
- Features engineered from user-confirmed data
- **GradientBoostingClassifier** predicts optimal sustainable material
- Returns: top 3 recommendations with confidence scores
- CO₂ comparison: current vs recommended
- Sustainability score (0–100) = weighted recyclability + biodegradability + inverse(CO₂) + inverse(cost)

---

## 📊 Dataset

- **1,470 rows** 
  - Missing values (~7% per column)
  - Duplicate rows (~40)
  - Mixed units (g/kg)
  - Text variations ("plastic bottle", "PET btl", "PLASTIC BOTTLE")
  - Outliers in CO₂ and cost columns
- **10 product categories**, **18 packaging materials**
- Target: `recommended_material` (18 classes)

---

## 🌍 CO₂ Formula

```
CO₂ = EmissionFactor × (WeightKg / 1000) × DistanceKm
```

Transport emission factors (kg CO₂ per tonne-km):
- Road:  0.096
- Rail:  0.028
- Sea:   0.010
- Air:   0.602

---

## 🤖 ML Model

| Parameter         | Value                |
|-------------------|----------------------|
| Algorithm         | GradientBoostingClassifier |
| n_estimators      | 200                  |
| learning_rate     | 0.1                  |
| max_depth         | 5                    |
| Features          | 10 engineered features |
| Target classes    | 18 materials         |
| Validation        | 5-fold cross-validation |

---

## 🎨 UI Features

- Dark luxury eco theme (deep forest greens + warm gold)
- Playfair Display + DM Sans typography
- Drag-and-drop image upload
- Real-time AI analysis display
- Editable confirmation form
- Radar, bar, doughnut, and scatter charts (Chart.js)
- Responsive for mobile + desktop
- Admin panel with material database

---

## 📝 Environment Variables

| Variable        | Description                        | Required |
|-----------------|------------------------------------|----------|
| `GEMINI_API_KEY`| Google Gemini API key              | Yes (for AI vision) |
| `SECRET_KEY`    | Flask session secret               | Optional |

---

## ⚡ Without Gemini API

The app works fully without a Gemini API key:
- OpenCV fallback provides shape/colour hints
- User manually fills in product details on the confirm page
- ML model runs normally on the confirmed inputs

---

*Built with Flask · scikit-learn · Gemini Vision · OpenCV · Chart.js*

