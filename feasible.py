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

# Derive output path from attribute
output_path = os.path.join("save", f"feasible_{attribute}.json")

# Load the full results
with open(input_path, "r") as f:
    data = json.load(f)

# Normalize inner keys to attribute-only (handle class-prefixed keys)
normalized = {}
for fname, attrs in data.items():
    # Extract the value for any key ending with _<attribute>
    val = next((v for k, v in attrs.items() if k.endswith(f"_{attribute}")), None)
    # If none found, fallback to direct attribute key
    if val is None:
        val = attrs.get(attribute, 0)
    normalized[fname] = {attribute: val}
# Replace data with normalized mapping
data = normalized

# Filter to only those entries with attribute == 1
feasible = {
    key: val
    for key, val in data.items()
    if val.get(attribute, 0) == 1
}

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Write filtered results
with open(output_path, "w") as f:
    json.dump(feasible, f, indent=4)

print(f"Kept {len(feasible)} entries where {attribute} == 1, saved to:\n  {output_path}")