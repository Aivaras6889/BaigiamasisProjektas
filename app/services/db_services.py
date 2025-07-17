from app.extensions import db

def add_commit(x):
    db.session.add(x)
    db.session.commit()
    return x 

def commit():
    db.session.commit()

def close_session():
    db.session.close()

def rollback():
    db.session.rollback