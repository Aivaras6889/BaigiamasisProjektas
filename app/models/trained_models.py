from datetime import datetime
from app.extensions import db

class TrainedModel(db.Model):
    """Trained ML models for prediction"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # 'svm', 'random_forest', 'neural_network'
    framework = db.Column(db.String(50), nullable=False)  # 'sklearn', 'tensorflow'
    file_path = db.Column(db.String(200), nullable=False)
    scaler_path = db.Column(db.String(200))
    
    # Performance metrics
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    
    # Training info
    training_samples = db.Column(db.Integer)
    training_time = db.Column(db.Float)
    hyperparameters = db.Column(db.Text)  # JSON string
    
    # Status
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)