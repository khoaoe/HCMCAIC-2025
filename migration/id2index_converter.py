import json
import re
import os

def convert_global_to_id2index(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        global_map = json.load(f)

    # Build id2index mapping with parsed path format
    id2index = {}
    
    for idx, path in enumerate(global_map.values()):
        # Parse the path to extract batch, video, and frame info
        # Expected format: .../L{batch}/.../V{video}/{frame}.webp
        
        # Extract batch number (L followed by digits)
        batch_match = re.search(r'/L(\d+)/', path)
        
        # Extract video number (V followed by digits)  
        video_match = re.search(r'/V(\d+)/', path)
        
        # Extract frame number from filename (digits before .webp)
        filename = os.path.basename(path)
        frame_match = re.search(r'(\d+)\.webp$', filename)
        
        if batch_match and video_match and frame_match:
            # Convert to integers to remove leading zeros, then back to strings
            batch_num = str(int(batch_match.group(1)))
            video_num = str(int(video_match.group(1)))
            frame_num = str(int(frame_match.group(1)))
            
            # Create the new format: "batch/video/frame"
            id2index[str(idx)] = f"{batch_num}/{video_num}/{frame_num}"
        else:
            # Fallback: use original path if parsing fails
            print(f"Warning: Could not parse path {path}")
            id2index[str(idx)] = path

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(id2index, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # get the parent directory
    parent_dir = os.path.dirname(current_dir)
    # get the resources directory
    resources_dir = os.path.join(parent_dir, "resources")
    # input path output path (global2imgpath.json is in embedding_keys subdirectory)
    input_path = os.path.join(resources_dir, "embedding_keys", "global2imgpath.json")
    output_path = os.path.join(resources_dir, "id2index.json")
    
    convert_global_to_id2index(
        input_path=input_path,
        output_path=output_path,
    )
    print("id2index.json created successfully.")