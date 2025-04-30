import json

def inspect_attributes(json_path):
    # Read the JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Get all top-level keys
    keys = list(data.keys())
    print(f"Total number of keys: {len(keys)}\n")

    # Iterate over each key and print its "word" list
    for key in keys:
        info = data[key]
        words = info.get("word")
        print(f"{key}:")
        if isinstance(words, list):
            print("  word =", words)
        else:
            print('  (no "word" field or it is not a list)')
        print()

if __name__ == "__main__":
    inspect_attributes("exampleData/celebA/attribute.json")