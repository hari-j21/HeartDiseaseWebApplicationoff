import os
import io
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

from flask import Flask, render_template, request, send_file, jsonify

# Fix for matplotlib in server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# DB
from flask_sqlalchemy import SQLAlchemy

# ------------------ APP SETUP ------------------ #
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ------------------ DATABASE MODEL ------------------ #
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Float)
    sex = db.Column(db.Float)
    cp = db.Column(db.Float)
    trestbps = db.Column(db.Float)
    chol = db.Column(db.Float)
    prediction = db.Column(db.String(50))

# ------------------ INIT DB ------------------ #
with app.app_context():
    db.create_all()

# ------------------ FILES ------------------ #
MODEL_FILE = "model.pkl"
DATA_FILE = "heart.csv"

# ------------------ TRAIN MODEL ------------------ #
def train_model():
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

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    return model

# ------------------ LOAD MODEL ------------------ #
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
else:
    model = train_model()

# ------------------ REPORT ------------------ #
def generate_report(data, risk, probability, confidence):
    file_path = "static/report.pdf"

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Heart Disease Report", styles['Title']))
    story.append(Spacer(1, 10))

    story.append(Paragraph(f"Risk: {risk}", styles['Normal']))
    story.append(Paragraph(f"Probability: {probability}%", styles['Normal']))
    story.append(Paragraph(f"Confidence: {confidence}%", styles['Normal']))

    doc = SimpleDocTemplate(file_path)
    doc.build(story)

    return file_path

# ------------------ ROUTES ------------------ #
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

        risk = "High Risk" if pred == 1 else "Low Risk"

        # Save to DB
        patient = Patient(
            age=features[0][0],
            sex=features[0][1],
            cp=features[0][2],
            trestbps=features[0][3],
            chol=features[0][4],
            prediction=risk
        )
        db.session.add(patient)
        db.session.commit()

        # Simple chart (instead of SHAP)
        fig, ax = plt.subplots()
        ax.bar(["Risk"], [probability])
        plot_path = "static/chart.png"
        fig.savefig(plot_path)
        plt.close()

        # PDF
        report_path = generate_report(request.form, risk, probability, confidence)

        return render_template(
            "result.html",
            risk=risk,
            probability=probability,
            confidence=confidence,
            chart=plot_path,
            report=report_path
        )

    except Exception as e:
        return str(e)

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

        return jsonify({
            "risk": "High Risk" if pred == 1 else "Low Risk",
            "probability": round(proba * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ------------------ RUN ------------------ #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))