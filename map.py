import json
import numpy as np
def get_original_indices_by_attribute(attribute_name, attr_assigned, attr_data, mapping_data, find_matching=False, save_path=None):
    """
    Get the original indices of samples with specific attribute value
    Args:
        attribute_name: str, the name of attribute in json file
        attr_assigned: str, the specific value of attribute
        attr_data: str or dict, path to attribute json file or loaded json data
        mapping_data: str or dict, path to mapping json file or loaded mapping dict
        find_matching: bool, if True find matching attributes, if False find non-matching (default: False)
        save_path: str, optional path to save indices to txt file
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
    print(f"Original indices (first 10): {original_indices[:10]}...")
    
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

# Example usage with file paths:
non_red_hair_indices = get_original_indices_by_attribute(  # Will find non-red hair by default
    "hair color", 
    "red",
    "exampleData/celebA/attribute.json",
    "exampleData/celebA/idx_mapping.json"
)

red_hair_indices = get_original_indices_by_attribute(  # Will find red hair
    "hair color", 
    "red",
    "exampleData/celebA/attribute.json",
    "exampleData/celebA/idx_mapping.json",
    find_matching=True
)

# Example usage with saving to file:
non_red_hair_indices = get_original_indices_by_attribute(
    "hair color", 
    "red",
    "exampleData/celebA/attribute.json",
    "exampleData/celebA/idx_mapping.json",
    save_path="exampleData/celebA/non_red_hair_indices.txt"
)
