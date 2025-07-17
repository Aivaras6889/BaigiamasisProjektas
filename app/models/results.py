from datetime import datetime
from flask_login import UserMixin
from app.extensions import db, bcrypt


class Results(db.Model):
    """Model for storing prediction results"""
    __tablename__ = 'results'

    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.Integer, db.ForeignKey('profiles.id'), nullable=False)
    prediction = db.Column(db.String(256), nullable=False)
    
    # Add these proper columns
    confidence = db.Column(db.Float, nullable=True)
    actual_class = db.Column(db.Integer, nullable=True)  
    model_name = db.Column(db.String(255), nullable=True)
    image_path = db.Column(db.String(500), nullable=True)
    prediction_time = db.Column(db.Float, nullable=True)  # Time taken to predict
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    profile = db.relationship('Profile', back_populates='results')