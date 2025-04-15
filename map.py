import json
import numpy as np
def get_original_indices_by_attribute(attribute_name, attr_assigned, attr_data, mapping_data, find_matching=False, save_path=None, num_display=10):
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

    # Load mapping data if path is provided
    if isinstance(mapping_data, str):
        try:
            with open(mapping_data, 'r') as f:
                mapping_dict = json.load(f)
                filtered_to_original = mapping_dict["filtered_to_original"]
        except Exception as e:
            print(f"Error loading mapping file: {e}")
            return []
    else:
        filtered_to_original = mapping_data.get("filtered_to_original", mapping_data)

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
    
    # Convert to original indices
    filtered_to_original = {int(k): v for k, v in filtered_to_original.items()}
    original_indices = [filtered_to_original[i] for i in filtered_indices if i in filtered_to_original]
    
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

# Example usage for finding red hair (commented out)
'''
red_hair_indices = get_original_indices_by_attribute(  # Will find red hair
    "hair color", 
    "red",
    "exampleData/celebA/attribute.json",
    "exampleData/celebA/idx_mapping.json",
    find_matching=True,
    num_display=100  # Display 100 indices
)

# Example usage with saving to file:
non_red_hair_indices = get_original_indices_by_attribute(
    "hair color", 
    "red",
    "exampleData/celebA/attribute.json",
    "exampleData/celebA/idx_mapping.json",
    find_matching=False,  # Finding hair color != red
    save_path="exampleData/celebA/non_red_hair_indices.txt",
    num_display=100  # Display 100 indices
)
'''

# Example usage for finding non-brown skin
non_brown_skin_indices = get_original_indices_by_attribute(
    "skin", 
    "brown",
    "exampleData/celebA/attribute.json",
    "exampleData/celebA/idx_mapping.json",
    find_matching=False,  # Finding skin != brown
    save_path="exampleData/celebA/non_brown_skin_indices.txt",
    num_display=100  # Display 100 indices
)

# Find intersection and differences between non_red_hair and non_brown_skin indices
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
    "exampleData/celebA/non_red_hair_indices.txt",
    "exampleData/celebA/non_brown_skin_indices.txt",
    "exampleData/celebA/non_red_hair_and_non_brown_skin.txt",  # Intersection (both conditions)
    "exampleData/celebA/only_non_red_hair.txt",                # Only in non_red_hair
    "exampleData/celebA/only_non_brown_skin.txt"               # Only in non_brown_skin
)
