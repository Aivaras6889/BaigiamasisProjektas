
from datetime import datetime
from app.extensions import db


class DatasetImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'))
    image_path = db.Column(db.String(255), nullable=False)
    class_id = db.Column(db.Integer, nullable=False)
    dataset_type = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    dataset = db.relationship('Dataset', back_populates='images')