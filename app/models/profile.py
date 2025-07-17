from datetime import datetime
from flask_login import UserMixin
from app.extensions import db, bcrypt


class Profile(UserMixin, db.Model):
    """User model for storing user related details"""
    __tablename__ = 'profiles'

    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(64))
    last_name = db.Column(db.String(64))
    birthday = db.Column(db.DateTime)
    signID = db.Column(db.Integer, db.ForeignKey('traffic_signs.id'), nullable=True)
    

    user= db.relationship('User', back_populates='profile')
    signs = db.relationship('TrafficSign', back_populates='user')  
    results = db.relationship('Results', back_populates='profile')


    @property
    def full_name(self):
        """Get user's full name"""
        return f"{self.first_name or ''} {self.last_name or ''}".strip() or self.username

    def __repr__(self):
        return f'<User {self.full_name}>'