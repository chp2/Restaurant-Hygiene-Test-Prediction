import json
import os

# Example data
data = [
    {"name": "A", "value": 30},
    {"name": "B", "value": 80},
    {"name": "C", "value": 45},
    {"name": "D", "value": 60}
]

# Convert the data to JSON format
json_data = json.dumps(data, indent=4)

# Specify the file path
file_path = 'data.json'

# Save the JSON data to a file
try:
    with open(file_path, 'w') as file:
        file.write(json_data)
    print(f"File saved successfully at {os.path.abspath(file_path)}")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")
