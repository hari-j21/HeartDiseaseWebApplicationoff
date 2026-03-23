from database.db import db
from datetime import datetime

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Float)
    sex = db.Column(db.Float)
    cp = db.Column(db.Float)
    trestbps = db.Column(db.Float)
    chol = db.Column(db.Float)
    probability = db.Column(db.Float)
    confidence = db.Column(db.Float)
    prediction = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)