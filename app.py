from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
import json
import os

app = Flask(__name__)
DATA_FILE = 'study_sessions.json'

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
    return render_template('studySessions.html', sessions=sessions)

@app.route('/health')
def health():
    return "OK", 200

if __name__ == '__main__':
    app.run(debug=True)
