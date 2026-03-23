from database.db import db
from datetime import datetime

class Patient(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    age = db.Column(db.Integer)
    sex = db.Column(db.Integer)
    cp = db.Column(db.Integer)

    trestbps = db.Column(db.Integer)
    chol = db.Column(db.Integer)

    prediction = db.Column(db.String(50))

    created_at = db.Column(db.DateTime, default=datetime.utcnow)