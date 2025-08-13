import json
import numpy as np
def get_original_indices_by_attribute(attribute_name, attr_assigned, attr_data, mapping_data=None, find_matching=False, save_path=None, num_display=10):
    """
    Get the original indices of samples with specific attribute value
    Args:
        attribute_name: str, the name of attribute in json file
        attr_assigned: str, the specific value of attribute
        attr_data: str or dict, path to attribute json file or loaded json data
        mapping_data: str or dict, path to mapping json file or loaded mapping dict
        find_matching: bool, if True find matching attributes, if False find non-matching (default: False)
        save_path: str, optional path to save indices to txt file
        num_display: int, number of sample indices to display (default: 10)
    Returns:
        list: original indices of samples with the specified attribute value
    """
    # Load attribute data if path is provided
    if isinstance(attr_data, str):
        try:
            with open(attr_data, 'r') as f:
                attr_json = json.load(f)
        except Exception as e:
            print(f"Error loading attribute file: {e}")
            return []
    else:
        attr_json = attr_data

    # Load mapping data if path is provided and exists, otherwise skip mapping
    if mapping_data is None:
        filtered_to_original = None
    elif isinstance(mapping_data, str):
        try:
            with open(mapping_data, 'r') as f:
                mapping_dict = json.load(f)
                filtered_to_original = mapping_dict.get("filtered_to_original")
        except FileNotFoundError:
            print(f"Mapping file {mapping_data} not found, skipping mapping.")
            filtered_to_original = None
        except Exception as e:
            print(f"Error loading mapping file: {e}")
            return []
    else:
        filtered_to_original = mapping_data.get("filtered_to_original")

    # Verify attribute exists
    if attribute_name not in attr_json:
        print(f"Error: Attribute '{attribute_name}' not found")
        return []

    # Find matching indices
    filtered_indices = []
    total_samples = len(attr_json[attribute_name]["data"])
    for i, description in enumerate(attr_json[attribute_name]["data"]):
        if (description == attr_assigned) == find_matching:  # Changed logic to use find_matching parameter
            filtered_indices.append(i)
    
    # Convert to original indices if mapping is provided, otherwise use filtered indices directly
    if filtered_to_original:
        filtered_to_original = {int(k): v for k, v in filtered_to_original.items()}
        original_indices = [filtered_to_original[i] for i in filtered_indices if i in filtered_to_original]
    else:
        original_indices = filtered_indices
    
    match_str = "=" if find_matching else "!="
    print(f"\n{attribute_name}{match_str}{attr_assigned} information:")
    print(f"Number of samples: {len(filtered_indices)}")
    print(f"Original indices (first {num_display}): {original_indices[:num_display]}...")
    
    # Save indices to file if save_path is provided
    if save_path is not None:
        try:
            with open(save_path, 'w') as f:
                for idx in original_indices:
                    f.write(f"{idx}\n")
            print(f"Saved indices to {save_path}")
        except Exception as e:
            print(f"Error saving indices to file: {e}")
    
    return original_indices

# Example usage for finding yellow hair (commented out)
# yellow_hair_indices = get_original_indices_by_attribute(  # Will find yellow hair
#     "hair color", 
#     "yellow",
#     "exampleData/celebA/attribute.json",
#     "exampleData/celebA/idx_mapping.json",
#     find_matching=True,
#     num_display=100  # Display 100 indices
# )

non_redhair_indices = get_original_indices_by_attribute(
    "hair color",
    "red",
    "exampleData/celebA/attribute.json",
    "exampleData/celebA/idx_mapping.json",
    find_matching=False,  # Finding hair color != red
    save_path="exampleData/celebA/non_redhair_indices.txt",
    num_display=100
)

# Example usage with saving to file:
# non_yellow_hair_indices = get_original_indices_by_attribute(
#     "hair color", 
#     "yellow",
#     "exampleData/celebA/attribute.json",
#     "exampleData/celebA/idx_mapping.json",
#     find_matching=False,  # Finding hair color != yellow
#     save_path="exampleData/celebA/non_yellowhair_indices.txt",
#     num_display=100  # Display 100 indices
# )

# Example usage for finding non-brown skin
# non_brown_skin_indices = get_original_indices_by_attribute(
#     "skin", 
#     "brown",
#     "exampleData/celebA/attribute.json",
#     "exampleData/celebA/idx_mapping.json",
#     find_matching=False,  # Finding skin != brown
#     save_path="exampleData/celebA/non_brownskin_indices.txt",
#     num_display=100  # Display 100 indices
# )

# Example usage for finding non-sad emotion
# non_sad_emotion_indices = get_original_indices_by_attribute(
#     "emotion",
#     "sad",
#     "exampleData/celebA/attribute.json",
#     "exampleData/celebA/idx_mapping.json",
#     find_matching=False,  # Finding emotion != sad
#     save_path="exampleData/celebA/non_sademotion_indices.txt",
#     num_display=100  # Display 100 indices
# )

# Find intersection and differences between non_redhair and non_brown_skin indices
def find_intersection_and_differences(file1_path, file2_path, intersection_path, only_in_file1_path, only_in_file2_path):
    """
    Find and save the intersection and differences between two sets of indices from text files
    
    Args:
        file1_path: Path to first indices text file
        file2_path: Path to second indices text file
        intersection_path: Path to save the intersection indices
        only_in_file1_path: Path to save indices that appear only in file1
        only_in_file2_path: Path to save indices that appear only in file2
    """
    # Load indices from files
    with open(file1_path, 'r') as f:
        set1 = set([int(line.strip()) for line in f if line.strip()])
    
    with open(file2_path, 'r') as f:
        set2 = set([int(line.strip()) for line in f if line.strip()])
    
    # Find intersection and differences
    intersection = set1.intersection(set2)
    only_in_set1 = set1 - set2
    only_in_set2 = set2 - set1
    
    # Save intersection
    with open(intersection_path, 'w') as f:
        for idx in sorted(intersection):
            f.write(f"{idx}\n")
    
    # Save only in file1
    with open(only_in_file1_path, 'w') as f:
        for idx in sorted(only_in_set1):
            f.write(f"{idx}\n")
    
    # Save only in file2
    with open(only_in_file2_path, 'w') as f:
        for idx in sorted(only_in_set2):
            f.write(f"{idx}\n")
    
    # Print summary
    file1_name = file1_path.split('/')[-1]
    file2_name = file2_path.split('/')[-1]
    print(f"\nComparison summary:")
    print(f"Total in {file1_name}: {len(set1)}")
    print(f"Total in {file2_name}: {len(set2)}")
    print(f"Intersection: {len(intersection)}")
    print(f"Only in {file1_name}: {len(only_in_set1)}")
    print(f"Only in {file2_name}: {len(only_in_set2)}")
    
    return intersection, only_in_set1, only_in_set2

# Execute the comparison
find_intersection_and_differences(
    "exampleData/celebA/non_redhair_indices.txt",
    "exampleData/celebA/non_brownskin_indices.txt",
    "exampleData/celebA/non_redhair_brownskin_indices.txt",  # Intersection (both conditions)
    "exampleData/celebA/only_non_redhair_indices.txt",                # Only in non_redhair
    "exampleData/celebA/only_non_brownskin_indices.txt"               # Only in non_brown_skin
)

# Compute intersections involving non-redhair, non-brown-skin, and non-sad-emotion
# Load the three sets
def load_indices(path):
    with open(path, 'r') as f:
        return set(int(line.strip()) for line in f if line.strip())

brown_skin_set = load_indices("exampleData/celebA/non_brownskin_indices.txt")
sad_emotion_set = load_indices("exampleData/celebA/non_sademotion_indices.txt")

# Replace red_hair_set with yellow_hair_set and update file path
yellow_hair_set = load_indices("exampleData/celebA/non_yellowhair_indices.txt")
brown_skin_set = load_indices("exampleData/celebA/non_brownskin_indices.txt")
sad_emotion_set = load_indices("exampleData/celebA/non_sademotion_indices.txt")

# # Three-way intersection
# intersection_all = yellow_hair_set & brown_skin_set & sad_emotion_set
# with open("exampleData/celebA/non_yellowhair_brownskin_sademotion_indices.txt", 'w') as f:
#     for idx in sorted(intersection_all):
#         f.write(f"{idx}\n")

# # Pairwise intersection: non_yellow_hair ∧ non_brown_skin
intersection_yellowhair_brownskin = yellow_hair_set & brown_skin_set
with open("exampleData/celebA/non_yellowhair_brownskin_indices.txt", 'w') as f:
    for idx in sorted(intersection_yellowhair_brownskin):
        f.write(f"{idx}\n")

# # Pairwise intersections
# intersection_yellowhair_sademotion = yellow_hair_set & sad_emotion_set
# with open("exampleData/celebA/non_yellowhair_sademotion_indices.txt", 'w') as f:
#     for idx in sorted(intersection_yellowhair_sademotion):
#         f.write(f"{idx}\n")

# intersection_brownskin_sademotion = brown_skin_set & sad_emotion_set
# with open("exampleData/celebA/non_brownskin_sademotion_indices.txt", 'w') as f:
#     for idx in sorted(intersection_brownskin_sademotion):
#         f.write(f"{idx}\n")

# # Print summary
# print("\nThree-way and additional two-way intersection summary:")
# print(f"non_yellow_hair ∧ non_brown_skin: {len(intersection_yellowhair_brownskin)} samples")
# print(f"non_yellow_hair ∧ non_brown_skin ∧ non_sad_emotion: {len(intersection_all)} samples")
# print(f"non_yellow_hair ∧ non_sad_emotion: {len(intersection_yellowhair_sademotion)} samples")
# print(f"non_brown_skin ∧ non_sad_emotion: {len(intersection_brownskin_sademotion)} samples")

# Example usage: find indices for samples that are non-redhair AND non-sad emotion

find_intersection_and_differences(
    "exampleData/celebA/non_redhair_indices.txt",
    "exampleData/celebA/non_sademotion_indices.txt",
    "exampleData/celebA/non_redhair_sademotion_indices.txt",
    "exampleData/celebA/only_non_redhair_indices.txt",
    "exampleData/celebA/only_non_sademotion_indices.txt"
)

# Compute three-way intersection: samples that are non-redhair, non-brownskin, AND non-sad emotion
redhair_set = load_indices("exampleData/celebA/non_redhair_indices.txt")
brownskin_set = load_indices("exampleData/celebA/non_brownskin_indices.txt")
sademotion_set = load_indices("exampleData/celebA/non_sademotion_indices.txt")

intersection_all = redhair_set & brownskin_set & sademotion_set
output_path = "exampleData/celebA/non_redhair_brownskin_sademotion_indices.txt"
with open(output_path, 'w') as f:
    for idx in sorted(intersection_all):
        f.write(f"{idx}\n")
print(f"Saved three-way intersection (non_redhair_brownskin_sademotion) to {output_path}, total samples: {len(intersection_all)}")

# Example usage: find samples with yellow hair AND brown skin AND sad emotion
yellow_hair_indices = get_original_indices_by_attribute(
    "hair color",
    "yellow",
    "exampleData/celebA/attribute.json",
    "exampleData/celebA/idx_mapping.json",
    find_matching=True,
    save_path="exampleData/celebA/yellowhair_indices.txt",
    num_display=100
)
brown_skin_indices = get_original_indices_by_attribute(
    "skin",
    "brown",
    "exampleData/celebA/attribute.json",
    "exampleData/celebA/idx_mapping.json",
    find_matching=True,
    save_path="exampleData/celebA/brownskin_indices.txt",
    num_display=100
)
sad_emotion_indices = get_original_indices_by_attribute(
    "emotion",
    "sad",
    "exampleData/celebA/attribute.json",
    "exampleData/celebA/idx_mapping.json",
    find_matching=True,
    save_path="exampleData/celebA/sademotion_indices.txt",
    num_display=100
)

# Compute three-way intersection for yellow hair, brown skin, and sad emotion
yellow_set = load_indices("exampleData/celebA/yellowhair_indices.txt")
brown_set = load_indices("exampleData/celebA/brownskin_indices.txt")
sad_set = load_indices("exampleData/celebA/sademotion_indices.txt")

intersection_yellow_brown_sad = yellow_set & brown_set & sad_set
output_path = "exampleData/celebA/yellowhair_brownskin_sademotion_indices.txt"
with open(output_path, 'w') as f:
    for idx in sorted(intersection_yellow_brown_sad):
        f.write(f"{idx}\n")
print(f"Saved three-way intersection (yellowhair_brownskin_sademotion) to {output_path}, total samples: {len(intersection_yellow_brown_sad)}")


with open("exampleData/celebA/non_yellowhair_sademotion_indices.txt","w") as f: f.write("\n".join(map(str,sorted(set(get_original_indices_by_attribute("hair color","yellow","exampleData/celebA/attribute.json","exampleData/celebA/idx_mapping.json",find_matching=False)) & set(get_original_indices_by_attribute("emotion","sad","exampleData/celebA/attribute.json","exampleData/celebA/idx_mapping.json",find_matching=True)))) )+"\n")

# Example usage for finding non-pink color in a 10-class attribute dataset without mapping
non_pinkcolor_indices = get_original_indices_by_attribute(
    "color",
    "pink",
    "attribute10.json",
    None,
    find_matching=False,
    save_path="non_pinkcolor_indices.txt",
    num_display=100
)


non_yellowcolor_indices = get_original_indices_by_attribute(
    "color",
    "yellow",
    "attribute10.json",
    None,
    find_matching=False,
    save_path="non_yellowcolor_indices.txt",
    num_display=10
)


non_fabrictexture_indices = get_original_indices_by_attribute(
    "texture",
    "fabric",
    "attribute10.json",
    None,
    find_matching=False,
    save_path="non_fabrictexture_indices.txt",
    num_display=10
)

non_orangecolor_indices = get_original_indices_by_attribute(
    "color",
    "orange",
    "attribute10.json",
    None,
    find_matching=False,
    save_path="non_orangecolor_indices.txt",
    num_display=10
)