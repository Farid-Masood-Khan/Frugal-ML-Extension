# Frugal ML Extension — Food Demand Forecasting & Spoilage Hazard Prediction

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FYP Asset](https://img.shields.io/badge/BSCS%20FYP-Frugal%20Platform-f97316.svg)](https://github.com/Farid-Masood-Khan/Frugal)

Machine Learning extension module for **Frugal: AI-powered food waste & donation management system** (BSCS FYP 2024).

---

## 📌 Context & Data Provenance
- **Connection to Frugal:** Bridges Farid's strongest verified engineering asset ([`Farid-Masood-Khan/Frugal`](https://github.com/Farid-Masood-Khan/Frugal)) into applied supervised machine learning.
- **Data Provenance Statement:** Trained on a **documented synthetic dataset** simulating food category, ambient temperature ($18^\circ\text{C} - 40^\circ\text{C}$), hours post-preparation, and transit logistics. Clearly documented and transparently reported.

---

## 📁 Repository Structure
```
Frugal-ML-Extension/
├── README.md
├── requirements.txt
├── data/
│   └── synthetic_frugal_food_data.csv
├── src/
│   └── demand_forecast.py    # Demand Regressor & Spoilage Hazard Classifier
└── models/
    ├── frugal_preprocessor.joblib
    ├── demand_regressor.joblib
    ├── spoilage_classifier.joblib
    └── frugal_metrics.json
```

## 🚀 Running the Project
```bash
pip install -r requirements.txt
python src/demand_forecast.py
```
