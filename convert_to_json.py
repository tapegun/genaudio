import json
from collections import defaultdict

def convert_dataset_to_json(input_file, output_file="dataset.json"):
    data = defaultdict(list)
    current_dir = None

    with open(input_file, "r") as f:
        for line in f:
            if line == "\n":
                continue
            line = line.strip()
            if line.endswith(":"):  # New directory
                current_dir = line[:-1]
            elif current_dir:  # File within the current directory
                data[current_dir].append(line)

    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Converted dataset to {output_file}")

convert_dataset_to_json("dataset.txt")