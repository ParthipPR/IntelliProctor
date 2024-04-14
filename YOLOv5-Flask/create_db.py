from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from app import app, db, studentdb

    
if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        # Create a test instance of the studentdb model
        test_student = studentdb(
            rollno=1,
            name="rose",
            password="password",
            gender="Female",
            malpractice=False,
            score = 50
        )

        # Add the test student to the session
        db.session.add(test_student)

        # Commit the changes to the database
        db.session.commit()