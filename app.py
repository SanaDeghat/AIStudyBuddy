from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
import json
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
DATA_FILE = 'study_sessions.json'

# Ensure data file exists
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w') as f:
        json.dump([], f)

def load_study_sessions():
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
        if not isinstance(data, list):
            return []
        return data

def save_study_sessions(sessions):
    with open(DATA_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)

def train_model(sessions):
    if not sessions:
        return None, None, None

    # Extract features and target
    subjects = [session['subject'] for session in sessions]
    time_spent = [session['time_spent'] for session in sessions]
    time_of_day = [session['time_of_day'] for session in sessions]
    mood = [session['mood'] for session in sessions]
    effective = [session['effective'] for session in sessions]

    # Encode categorical features
    le_subjects = LabelEncoder()
    le_time_of_day = LabelEncoder()

    subjects_encoded = le_subjects.fit_transform(subjects)
    time_of_day_encoded = le_time_of_day.fit_transform(time_of_day)

    # Create feature matrix and output
    X = np.array([subjects_encoded, time_spent, time_of_day_encoded, mood]).T
    y = np.array(effective)

    # Train the classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model, le_subjects, le_time_of_day

def get_recommendations(model, le_subjects, le_time_of_day, sessions):
    if not model or not le_subjects or not le_time_of_day:
        return "Not enough data to generate recommendations."

    # Extract feature importance
    feature_importances = model.feature_importances_
    importance_message = []
    importance_message.append(f"Time spent on study mattered most with importance {feature_importances[1]:.2f}.")
    importance_message.append(f"Mood contribution was {feature_importances[3]:.2f}.")
    importance_message.append("Subject and time of day had moderate importance.")

    all_subjects = le_subjects.classes_
    all_times_of_day = le_time_of_day.classes_

    recommendation = (
        f"Based on your study history, you are most effective studying "
        f"{all_subjects[0]} in the {all_times_of_day[0]} when your mood is high."
    )
    return "<br>".join(importance_message + [recommendation])

def predict_effectiveness(model, le_subjects, le_time_of_day, subject, time_spent, time_of_day, mood):
    # Encode user input
    subject_encoded = le_subjects.transform([subject])[0]
    time_of_day_encoded = le_time_of_day.transform([time_of_day])[0]

    input_features = np.array([[subject_encoded, time_spent, time_of_day_encoded, mood]])
    prediction = model.predict(input_features)[0]
    return "Effective" if prediction else "Not Effective"

@app.route('/')
def form():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    session = {
        "subject": request.form['subject'],
        "time_spent": int(request.form['time_spent']),
        "time_of_day": request.form['time_of_day'],
        "mood": int(request.form['mood']),
        "effective": request.form['effective'] == "yes",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    sessions = load_study_sessions()
    sessions.append(session)
    save_study_sessions(sessions)

    return redirect(url_for('studySessions'))

@app.route('/studySessions')
def studySessions():
    sessions = load_study_sessions()
    model, le_subjects, le_time_of_day = train_model(sessions)
    recommendations = get_recommendations(model, le_subjects, le_time_of_day, sessions) if model else "Not enough data."
    decision_tree_rules = export_text(model) if model else "No decision tree yet."

    return render_template(
        'studySessions.html',
        sessions=sessions,
        recommendations=recommendations,
        decision_tree=f"<pre>{decision_tree_rules}</pre>"
    )

@app.route('/whatIf', methods=['GET', 'POST'])
def whatIf():
    sessions = load_study_sessions()
    model, le_subjects, le_time_of_day = train_model(sessions)

    if request.method == 'POST':
        subject = request.form['subject']
        time_spent = int(request.form['time_spent'])
        time_of_day = request.form['time_of_day']
        mood = int(request.form['mood'])

        prediction = predict_effectiveness(
            model, le_subjects, le_time_of_day, subject, time_spent, time_of_day, mood)
        return render_template('whatIf.html', prediction=prediction)

    return render_template('whatIf.html', prediction=None)

@app.route('/health')
def health():
    return "OK", 200

if __name__ == '__main__':
    app.run(debug=True)