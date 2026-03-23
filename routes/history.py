from flask import Blueprint, render_template, jsonify, request
import numpy as np
from datetime import datetime
from models.patient import Patient

history_bp = Blueprint("history", __name__)

@history_bp.route("/history")
def history():
    patients = Patient.query.order_by(Patient.created_at.desc()).all()
    return render_template("history.html", patients=patients)

# history.py only serves patient history view now.

