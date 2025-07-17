from app.extensions import db


class ModelResult(db.Model):
    table_name = 'model_results'
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(255), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    training_time = db.Column(db.Float, nullable=False)
    hyperparameters = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<ModelResult {self.id} - {self.model_name} ({self.model_type})>'