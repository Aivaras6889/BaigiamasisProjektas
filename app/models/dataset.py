from datetime import datetime
from app.extensions import db


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), default='loading')
    total_images = db.Column(db.Integer, default=0)
    num_classes = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    images = db.relationship('DatasetImage', back_populates='dataset', lazy=True)
