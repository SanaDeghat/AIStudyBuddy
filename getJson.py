import json
import random

OUTPUT_FILE = "study_sessions.json"
NUM_ENTRIES = 200 

subjects = [
    "Math", "English", "Biology", "Chemistry",
    "Physics", "Socials", "Computer Science"
]

times_of_day = ["Morning", "Afternoon", "Evening", "Night"]
moods = ["Low", "Medium", "High"]

def random_duration():
    return random.choice([
        random.randint(10, 30),
        random.randint(31, 60),
        random.randint(61, 90),
        random.randint(91, 180)
    ])

def generate_effectiveness(subject, duration, mood, time_of_day):
    score = 0

    if duration >= 60:
        score += 1
    if mood == "High":
        score += 1
    if time_of_day in ["Morning", "Afternoon"]:
        score += 1
    if subject in ["Math", "Physics"] and duration < 30:
        score -= 1

    return "Yes" if score >= 2 else "No"


data = []

for _ in range(NUM_ENTRIES):
    subject = random.choice(subjects)
    duration = random_duration()
    time_of_day = random.choice(times_of_day)
    mood = random.choice(moods)
    effectiveness = generate_effectiveness(subject, duration, mood, time_of_day)

    data.append({
        "Subject": subject,
        "Duration": duration,
        "TimeOfDay": time_of_day,
        "Mood": mood,
        "Effectiveness": effectiveness
    })

with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f, indent=4)

print(f"Generated {NUM_ENTRIES} study sessions → {OUTPUT_FILE}")
