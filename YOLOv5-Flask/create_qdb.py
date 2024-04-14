from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from app import app, db, questiondb

if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        # List of sample questions and answers
        sample_questions = [
    {"qno": 1, "question": "What is the capital of France?", "answer": "Paris"},
    {"qno": 2, "question": "Who is the author of 'To Kill a Mockingbird'?", "answer": "Harper Lee"},
    {"qno": 3, "question": "What is the chemical symbol for water?", "answer": "H2O"},
    {"qno": 4, "question": "What year did World War II end?", "answer": "1945"},
    {"qno": 5, "question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"qno": 6, "question": "What is the tallest mountain in the world?", "answer": "Mount Everest"},
    {"qno": 7, "question": "Who wrote '1984'?", "answer": "George Orwell"},
    {"qno": 8, "question": "What is the currency of Japan?", "answer": "Yen"},
    {"qno": 9, "question": "Who is known as the father of modern physics?", "answer": "Albert Einstein"},
    {"qno": 10, "question": "What is the chemical formula for table salt?", "answer": "NaCl"},
    {"qno": 11, "question": "What is the square root of 144?", "answer": "12"},
    {"qno": 12, "question": "Who wrote 'Hamlet'?", "answer": "William Shakespeare"},
    {"qno": 13, "question": "Which planet is known as the Red Planet?", "answer": "Mars"},
    {"qno": 14, "question": "What is the freezing point of water in Celsius?", "answer": "0"},
    {"qno": 15, "question": "Who is credited with the invention of the telephone?", "answer": "Alexander Graham Bell"},
    {"qno": 16, "question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"qno": 17, "question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    {"qno": 18, "question": "What is the largest organ in the human body?", "answer": "Skin"},
    {"qno": 19, "question": "Who painted 'Starry Night'?", "answer": "Vincent van Gogh"},
    {"qno": 20, "question": "What is the boiling point of water in Fahrenheit?", "answer": "212"},
    # Add more questions here
]


        # Add each question to the database
        for question_data in sample_questions:
            question = questiondb(
                qno=question_data["qno"],
                question=question_data["question"],
                answer=question_data["answer"]
            )
            db.session.add(question)

        # Commit the changes to the database
        db.session.commit()
