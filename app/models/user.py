from datetime import datetime
from flask_login import UserMixin
from app.extensions import db, bcrypt


class User(UserMixin, db.Model):
    """User model for storing user related details"""
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    last_login = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    profileID = db.Column(db.Integer, db.ForeignKey('profiles.id'), nullable=False)
    profile = db.relationship('Profile', back_populates='user')

    def set_password(self, password):
        """Set user's password"""
        self.password_hash = bcrypt.generate_password_hash(password)
    def check_password(self, password):
        """Check user's password"""
        return bcrypt.check_password_hash(self.password_hash, password)
    def __repr__(self):
        return f'<User {self.username}>'
    