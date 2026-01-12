import json
import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Configuration for dataset
n_records = 10000  # Specify the number of records
subjects = ['Math', 'Science', 'History', 'Literature', 'Art']
moods = ['Happy', 'Neutral', 'Sad', 'Stressed', 'Motivated']
time_of_day = ['Morning', 'Afternoon', 'Evening', 'Night']

# Generate dataset
data = []
for _ in range(n_records):
    record = {
        'subject': random.choice(subjects),
        'time_spent': np.random.randint(1, 181),  # Random time spent in minutes
        'mood': random.choice(moods),
        'effective': random.choice([True, False]),
        'time_of_day': random.choice(time_of_day),
        'timestamp': fake.date_time_this_year().isoformat()  # Random timestamp
    }
    data.append(record)

# Save dataset to a JSON file
with open('large_student_productivity_data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

# Show an example of the generated data
print(json.dumps(data[:5], indent=4))