import json
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Filter feasible images based on attribute results JSON.")
parser.add_argument('--input_path', type=str, required=True, help='Path to the results JSON file')
args = parser.parse_args()

# Use arguments
input_path = args.input_path

# Derive attribute name from input file prefix (e.g., 'pinkcolor' from 'pinkcolor_results_*.json')
attribute = os.path.basename(input_path).split('_results')[0]

# Support multiple attributes separated by underscores
attr_list = attribute.split('_')

# Derive output path from attribute
output_path = os.path.join("save", f"feasible_{attribute}.json")

# Load the full results
with open(input_path, "r") as f:
    data = json.load(f)

# Normalize inner keys to only include specified attributes
normalized = {}
for fname, attrs in data.items():
    # Build a dict with each attribute's value (default to 0 if missing)
    normalized[fname] = {attr: attrs.get(attr, 0) for attr in attr_list}
data = normalized

# Filter entries where all specified attributes equal 1
feasible = {
    key: val
    for key, val in data.items()
    if all(val.get(attr, 0) == 1 for attr in attr_list)
}

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Write filtered results
with open(output_path, "w") as f:
    json.dump(feasible, f, indent=4)

print(f"Kept {len(feasible)} entries where {attribute} == 1, saved to:\n  {output_path}")