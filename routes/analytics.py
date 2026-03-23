from flask import Blueprint, render_template, jsonify, request
from models.patient import Patient
from database.db import db
import statistics
from datetime import datetime

analytics_bp = Blueprint("analytics", __name__)


@analytics_bp.route("/analytics")
def analytics_page():
    patients = Patient.query.all()
    total = len(patients)

    high_count = sum(1 for p in patients if p.prediction == "High Risk")
    low_count = sum(1 for p in patients if p.prediction == "Low Risk")

    ages = [p.age for p in patients if p.age is not None]
    chol = [p.chol for p in patients if p.chol is not None]
    trestbps = [p.trestbps for p in patients if p.trestbps is not None]

    avg_age = round(statistics.mean(ages), 1) if ages else 0
    median_age = round(statistics.median(ages), 1) if ages else 0
    avg_chol = round(statistics.mean(chol), 1) if chol else 0
    avg_trestbps = round(statistics.mean(trestbps), 1) if trestbps else 0

    high_risk_pct = round((high_count / total) * 100, 1) if total else 0

    return render_template(
        "analytics.html",
        total=total,
        high_count=high_count,
        low_count=low_count,
        high_risk_pct=high_risk_pct,
        avg_age=avg_age,
        median_age=median_age,
        avg_chol=avg_chol,
        avg_trestbps=avg_trestbps,
        age_labels=[],
        age_values=[],
        sex_labels=[],
        sex_values=[],
        cp_labels=[],
        cp_values=[],
        history_dates=[],
        predictions_over_time=[],
        recent_predictions=[]
    )


@analytics_bp.route("/api/analytics")
def analytics_api():
    patients = Patient.query.all()
    total = len(patients)

    high_count = sum(1 for p in patients if p.prediction == "High Risk")
    low_count = sum(1 for p in patients if p.prediction == "Low Risk")

    ages = [p.age for p in patients if p.age is not None]
    chol = [p.chol for p in patients if p.chol is not None]

    avg_age = round(sum(ages) / len(ages), 1) if ages else 0
    median_age = sorted(ages)[len(ages) // 2] if ages else 0
    avg_chol = round(sum(chol) / len(chol), 1) if chol else 0

    high_risk_pct = round((high_count / total) * 100, 1) if total else 0

    age_groups = {"30-40": 0, "41-50": 0, "51-60": 0, "61+": 0}
    for p in patients:
        if p.age is None:
            continue
        if p.age <= 40:
            age_groups["30-40"] += 1
        elif p.age <= 50:
            age_groups["41-50"] += 1
        elif p.age <= 60:
            age_groups["51-60"] += 1
        else:
            age_groups["61+"] += 1

    sex_counts = {"Male": 0, "Female": 0}
    for p in patients:
        if p.sex == 1:
            sex_counts["Male"] += 1
        else:
            sex_counts["Female"] += 1

    cp_counts = {}
    for p in patients:
        cp = p.cp if p.cp is not None else "unknown"
        cp_counts[cp] = cp_counts.get(cp, 0) + 1

    history_dates = [p.created_at.strftime("%Y-%m-%d") for p in patients if p.created_at]
    predictions_over_time = [1 if p.prediction == "High Risk" else 0 for p in patients]

    return jsonify({
        "total": total,
        "high_count": high_count,
        "low_count": low_count,
        "high_risk_pct": high_risk_pct,
        "avg_age": avg_age,
        "median_age": median_age,
        "avg_chol": avg_chol,
        "age_labels": list(age_groups.keys()),
        "age_values": list(age_groups.values()),
        "sex_labels": list(sex_counts.keys()),
        "sex_values": list(sex_counts.values()),
        "cp_labels": list(cp_counts.keys()),
        "cp_values": list(cp_counts.values()),
        "history_dates": history_dates,
        "predictions_over_time": predictions_over_time
    })
