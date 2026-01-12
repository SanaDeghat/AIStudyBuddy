import json
import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
DATA_FILE = 'study_sessions.json'

def load_data():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_data(data):
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_duration_range(minutes):
    minutes = int(minutes)
    if minutes <= 30: return "0-30"
    elif minutes <= 60: return "31-60"
    elif minutes <= 90: return "61-90"
    else: return "90+"

def train_model(sessions):

    if len(sessions) < 5: 
        return None, None, None, None

    df = pd.DataFrame(sessions)
    
    df['DurationRange'] = df['Duration'].apply(get_duration_range)

    encoders = {}
    for col in ['Subject', 'TimeOfDay', 'Mood', 'DurationRange', 'Effectiveness']:
        le = LabelEncoder()
        df[col + '_Encoded'] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = ['Subject_Encoded', 'DurationRange_Encoded', 'TimeOfDay_Encoded', 'Mood_Encoded']
    X = df[feature_cols]
    y = df['Effectiveness_Encoded']

    clf = DecisionTreeClassifier(random_state=67)
    clf.fit(X, y)

    return clf, encoders, feature_cols, df


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        new_entry = {
            'Subject': request.form['subject'],
            'Duration': int(request.form['duration']),
            'TimeOfDay': request.form['time_of_day'],
            'Mood': request.form['mood'],
            'Effectiveness': request.form['effectiveness']
        }
        data = load_data()
        data.append(new_entry)
        save_data(data)
        return redirect(url_for('history'))
    
    return render_template('index.html')

@app.route('/history')
def history():
    data = load_data()
    return render_template('history.html', sessions=data)

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    data = load_data()
    model, encoders, feature_names_encoded, df = train_model(data)

    if not model:
        return render_template('recommendations.html', error="Not enough data to train model. Log at least 5 sessions.")

    importances = model.feature_importances_
    feature_names_readable = ['Subject', 'Duration', 'Time of Day', 'Mood'] 
    importance_dict = dict(zip(feature_names_readable, importances))
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

    tree_text = export_text(model, feature_names=feature_names_readable)

    subjects = encoders['Subject'].classes_
    times = encoders['TimeOfDay'].classes_
    moods = encoders['Mood'].classes_
    durations = encoders['DurationRange'].classes_
    
    best_combo = None
    best_prob = -1

    for s in subjects:
        for t in times:
            for m in moods:
                for d in durations:
                    s_enc = encoders['Subject'].transform([s])[0]
                    t_enc = encoders['TimeOfDay'].transform([t])[0]
                    m_enc = encoders['Mood'].transform([m])[0]
                    d_enc = encoders['DurationRange'].transform([d])[0]
                    
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba([[s_enc, d_enc, t_enc, m_enc]])
                        # Find index of 'Yes'
                        yes_index = list(encoders['Effectiveness'].classes_).index('Yes')
                        # Check if model has seen enough classes to output probability for Yes
                        if len(probs[0]) > yes_index:
                             prob_success = probs[0][yes_index]
                        else:
                             # If model only knows 'No', prob of Yes is 0
                             prob_success = 0.0
                    else:
                        prob_success = 0.0

                    if prob_success > best_prob:
                        best_prob = prob_success
                        best_combo = f"Subject: {s}, Time: {t}, Mood: {m}, Duration: {d}"

    prediction_result = None
    prediction_prob = None

    if request.method == 'POST':
        input_sub = request.form['subject']
        input_time = request.form['time_of_day']
        input_mood = request.form['mood']
        input_dur = get_duration_range(request.form.get('duration', 30)) 

        try:
            s_enc = encoders['Subject'].transform([input_sub])[0]
            t_enc = encoders['TimeOfDay'].transform([input_time])[0]
            m_enc = encoders['Mood'].transform([input_mood])[0]
            d_enc = encoders['DurationRange'].transform([input_dur])[0]

            probs = model.predict_proba([[s_enc, d_enc, t_enc, m_enc]])[0]
            
            max_index = np.argmax(probs)
            prediction_label = encoders['Effectiveness'].classes_[max_index]
            prediction_prob = probs[max_index-2] 

            prediction_result = prediction_label
        except ValueError:
            prediction_result = "Error: Input value never seen in training data."

    return render_template('recommendations.html', 
                           importance=sorted_importance,
                           tree_text=tree_text,
                           best_recommendation=best_combo,
                           best_prob=best_prob,
                           prediction_result=prediction_result,
                           prediction_prob=prediction_prob)

if __name__ == '__main__':
    app.run(debug=True)