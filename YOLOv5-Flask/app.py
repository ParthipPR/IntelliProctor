from flask import Flask, render_template, request, redirect, jsonify, url_for, flash
import subprocess
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///student.db'

db = SQLAlchemy(app)

class studentdb(db.Model):
    rollno = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(200), nullable = False)
    password = db.Column(db.String(200), nullable = False)
    gender = db.Column(db.String(200), nullable = False)
    malpractice = db.Column(db.Boolean, default = False)
    score = db.Column(db.Integer, default = False)

class questiondb(db.Model):
    qno = db.Column(db.Integer, primary_key = True)
    question = db.Column(db.String(500),nullable = False)
    answer = db.Column(db.String(500),nullable = False)



@app.route("/")
def index():
    return render_template('login_improved.html')

@app.route("/teacherpage")
def teacherpage():
    students = studentdb.query.order_by(studentdb.rollno).all()
    questions = questiondb.query.order_by(questiondb.qno).all()
    return render_template('teacherpage.html', students = students,questions = questions)

@app.route('/exampage',methods=['POST','GET'])
def studentpage():
    return render_template('Exampage.html')

@app.route("/studentlogin", methods=['POST','GET'])
def stu_login():
    if request.method == "POST":
        rollno = request.form.get("rollno")
        password = request.form.get("password")
        student = studentdb.query.filter_by(rollno=rollno).first()
        questions = questiondb.query.order_by(questiondb.qno).all()

        if student:
            if password == student.password:
                # Password is correct, redirect to exam page
                opencam(rollno,questions)
                return render_template('Exampage.html', rollno = rollno, questions = questions)
            else:
                # Password is incorrect, flash error message and redirect back to login page
                flash("Incorrect password. Please try again.", "error")
                return redirect('/studentlogin')
        else:
            # Student record not found, flash error message and redirect back to login page
            flash("Invalid roll number. Please try again.", "error")
            return redirect('/studentlogin')
    return redirect('/')

@app.route("/teacherlogin", methods=["GET", "POST"])
def teacher_login():
    if request.method == "POST":
        
        username = request.form.get("username")
        password = request.form.get("password")
        
        if username == "admin" and password == "admin":
            # Redirect to a dashboard or profile page upon successful login
            return redirect('/teacherpage')
        else:
            # Display an error message if credentials are incorrect
            return render_template("login_improved.html", error="Invalid username or password")
    else:
        # Render the login form
        return redirect('/')


@app.route('/add', methods = ['POST', 'GET'])
def add():
    if request.method == 'POST':
        rollno = request.form['rollno'].strip()
        name = request.form['name'].strip()
        password = request.form['password'].strip()
        gender = request.form['gender'].strip()

        new_student = studentdb(rollno = rollno, name = name, password = password, gender = gender)

        try:
            db.session.add(new_student)
            db.session.commit()
            return redirect('/teacherpage')
        except:
            return 'there is an error in adding the student'

@app.route('/delete/<int:rollno>')
def delete(rollno):
    student_to_delete = studentdb.query.get_or_404(rollno)

    try:
        db.session.delete(student_to_delete)
        db.session.commit()
        return redirect('/teacherpage')
    except:
        return 'there is an error in deleting the student data'

@app.route("/update/<int:rollno>", methods = ['POST', 'GET'])
def update(rollno):
    student = studentdb.query.get_or_404(rollno)
    if request.method == 'POST':
        student.name = request.form['name']
        student.password = request.form['password']
        student.gender = request.form['gender']

        try:
            db.session.commit()
            return redirect('/teacherpage')
        except:
            return 'there is an error in updating the student data'
    else:
        return render_template ('update.html', student = student)

process = None

def opencam(rollno,questions):
    global process
    print("Opening camera...")
    if process is None or process.poll() is not None:
        process = subprocess.Popen(['python3', 'detect_test.py', '--source', '0', '--rollno',str(rollno)])
        print("Camera opened successfully")
    else:
        print("Camera is already open")
    return render_template('Exampage.html', rollno = rollno, questions = questions)

@app.route("/closecam", methods=['GET'])
def closecam():
    global process
    print("Closing camera...")
    if process and process.poll() is None:
        process.terminate()
        print("Camera closed successfully")
    else:
        print("No camera is currently open")
    return redirect('/')

@app.route('/add_question', methods = ['POST', 'GET'])
def add_question():
    if request.method == 'POST':
        qno =questiondb.query.count()
        question = request.form['QUESTION'].strip()
        answer = request.form['ANSWER'].strip()

        new_question = questiondb(qno = qno + 1, question = question, answer = answer)

        try:
            db.session.add(new_question)
            db.session.commit()
            return redirect('/teacherpage')
        except:
            return 'there is an error in adding the question'
        
@app.route('/delete_question/<int:qno>')
def delete_question(qno):
    question_to_delete = questiondb.query.get_or_404(qno)

    try:
        db.session.delete(question_to_delete)
        db.session.commit()
        return redirect('/teacherpage')
    except:
        return 'there is an error in deleting the question'
    
@app.route("/submit_answers/<int:rollno>", methods=["POST"])
def submit_answers(rollno):
    submitted_answers = request.form.getlist("answer")
    correct_answers = [question.answer for question in questiondb.query.all()]

    print("Submitted answers:", submitted_answers)
    print("Correct answers:", correct_answers)

    score = sum(submitted_answer == correct_answer for submitted_answer, correct_answer in zip(submitted_answers, correct_answers))
    total_questions = len(correct_answers)
    percentage_score = (score / total_questions) * 100
    print("Score:", score)
    print("Total questions:", total_questions)
    print("Percentage score:", percentage_score)

    student = studentdb.query.filter_by(rollno=rollno).first()

    if student:
        student.score =  percentage_score
        db.session.commit()
        closecam()
        return render_template("login_improved.html")
    else:
        return jsonify({"error": "Student not found."}), 404
        
if __name__ == "__main__":
    app.run(debug = False)  
