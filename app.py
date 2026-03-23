from flask import Flask, render_template, request, send_file, jsonify
import numpy as np
import pickle
import os
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from datetime import datetime
import io
import logging

from database.db import db
from models.patient import Patient

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

logging.basicConfig(
    filename="app.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    db.create_all()

MODEL_FILE = "model.pkl"
DATA_FILE = "heart.csv"

def train_model():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError("heart.csv not found")

    df = pd.read_csv(DATA_FILE)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    with open("model_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}")

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
else:
    model = train_model()

explainer = shap.TreeExplainer(model.named_steps["rf"])

def generate_report(patient_data, risk, probability, confidence):
    file_path = f"static/report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI CardioCare Medical Report", styles['Title']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%d-%m-%Y')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Prediction Results", styles['Heading2']))
    story.append(Paragraph(f"Risk Level: {risk}", styles['Normal']))
    story.append(Paragraph(f"Probability: {probability}%", styles['Normal']))
    story.append(Paragraph(f"Confidence: {confidence}%", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    for key, value in patient_data.items():
        story.append(Paragraph(f"{key}: {value}", styles['Normal']))

    doc = SimpleDocTemplate(file_path)
    doc.build(story)

    return file_path

def generate_insights(data, probability, confidence):
    insights = []

    if probability >= 75:
        insights.append("HIGH RISK - Consult doctor immediately")
    elif probability >= 50:
        insights.append("MODERATE RISK - Medical check recommended")
    else:
        insights.append("LOW RISK - Maintain healthy lifestyle")

    if float(data['chol']) > 240:
        insights.append("High cholesterol detected")

    if float(data['trestbps']) > 140:
        insights.append("High blood pressure detected")

    return insights

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = np.array([[
            float(request.form["age"]),
            float(request.form["sex"]),
            float(request.form["cp"]),
            float(request.form["trestbps"]),
            float(request.form["chol"]),
            float(request.form["fbs"]),
            float(request.form["restecg"]),
            float(request.form["thalach"]),
            float(request.form["exang"]),
            float(request.form["oldpeak"]),
            float(request.form["slope"]),
            float(request.form["ca"]),
            float(request.form["thal"])
        ]])

        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        probability = round(proba * 100, 2)
        confidence = round(max(proba, 1 - proba) * 100, 2)

        if probability >= 75:
            risk = "High Risk"
            badge = "danger"
        elif probability >= 50:
            risk = "Moderate Risk"
            badge = "warning"
        else:
            risk = "Low Risk"
            badge = "success"

        patient = Patient(
            age=float(request.form["age"]),
            sex=float(request.form["sex"]),
            cp=float(request.form["cp"]),
            trestbps=float(request.form["trestbps"]),
            chol=float(request.form["chol"]),
            probability=probability,
            confidence=confidence,
            prediction=risk
        )

        db.session.add(patient)
        db.session.commit()

        plot_path = f"static/explanation_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"

        shap_values = explainer.shap_values(features)
        importance = np.abs(shap_values.values[0])

        plt.figure(figsize=(10,6))
        plt.barh(range(len(importance)), importance)
        plt.yticks(range(len(importance)))
        plt.savefig(plot_path)
        plt.close()

        report_path = generate_report(request.form.to_dict(), risk, probability, confidence)
        insights = generate_insights(request.form, probability, confidence)

        return render_template(
            "result.html",
            risk=risk,
            badge=badge,
            probability=probability,
            confidence=confidence,
            shap_plot=plot_path,
            report_path=report_path,
            insights=insights
        )

    except Exception as e:
        logging.error(str(e))
        return render_template("error.html", error="Something went wrong")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        features = np.array([[
            float(data["age"]), float(data["sex"]), float(data["cp"]),
            float(data["trestbps"]), float(data["chol"]), float(data["fbs"]),
            float(data["restecg"]), float(data["thalach"]), float(data["exang"]),
            float(data["oldpeak"]), float(data["slope"]), float(data["ca"]),
            float(data["thal"])
        ]])

        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0][1]

        probability = round(proba * 100, 2)
        confidence = round(max(proba, 1 - proba) * 100, 2)

        risk = "High Risk" if probability >= 50 else "Low Risk"

        return jsonify({
            "risk": risk,
            "probability": probability,
            "confidence": confidence,
            "prediction": int(pred),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=False)