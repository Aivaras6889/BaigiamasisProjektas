from app.extensions import db

class TrafficSign(db.Model):
    __tablename__ = 'traffic_signs'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    class_id = db.Column(db.Integer, nullable=False)
    width = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Integer, nullable=False)
    roi_x1 = db.Column(db.Integer, nullable=True)
    roi_y1 = db.Column(db.Integer, nullable=True)
    roi_x2 = db.Column(db.Integer, nullable=True)
    roi_y2 = db.Column(db.Integer, nullable=True)
    is_train = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
  


    hog_features = db.Column(db.LargeBinary, nullable=True)
    haar_features = db.Column(db.LargeBinary, nullable=True)
    hue_histogram = db.Column(db.LargeBinary, nullable=True)
    
    user = db.relationship('Profile', back_populates='signs') 
    
    def __repr__(self):
        return f'<TrafficSign {self.id} - {self.filename} (Class ID: {self.class_id})>'