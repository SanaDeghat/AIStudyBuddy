import os
from flask import Flask , render_template,  request, redirect, url_for
import json
from datetime import datetime
import os

app = Flask(__name__)
dateFile= "study_sessions.json"

def load_file():
    if os.path.exists(dateFile):
        with open(dateFile, 'r') as f:
            return json.load(f)
    return []

def save_sessions(sessions):
    with open(dateFile, 'w') as f:
        json.dump(sessions, f)  

@app.route('/')
def index():
    if request.method == 'POST':
        sessions = load_file()
        new_session = {
            "subject": request.form["subject"],
            "time_spent": int(request.form["time_spent"]),
            "time_of_day": request.form["time_of_day"],
            "mood": request.form["mood"],
            "effective": request.form["effective"] == "yes",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        sessions.append(new_session)
        save_sessions(sessions)
        return redirect (url_for('history'))
    return render_template('index.html')
@app.route('/history')
def history():
    sessions = load_file()
    return render_template('history.html', sessions=sessions)


if __name__ == "__main__":
    app.run(debug=True)