import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple
import argparse
from tqdm import tqdm

# Add parent directory to path for imports
ROOT_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, ROOT_FOLDER)

from migration.keyframe_migration import migrate_keyframes


def parse_video_key_from_filename(filename: str) -> Tuple[int, int]:
    """
    Parse group and video numbers from filename like 'L21_V001.csv'
    
    Args:
        filename: CSV filename (e.g., 'L21_V001.csv')
    
    Returns:
        tuple: (group_num, video_num)
    """
    # Remove .csv extension
    name = filename.replace('.csv', '')
    
    # Parse L{group}_V{video} format
    if '_V' in name:
        parts = name.split('_V')
        if len(parts) == 2:
            group_part = parts[0]
            video_part = parts[1]
            
            if group_part.startswith('L'):
                group_num = int(group_part[1:])
                video_num = int(video_part)
                return group_num, video_num
    
    raise ValueError(f"Could not parse video key from filename: {filename}")


def csv_to_keyframe_mapping(csv_file_path: str, global_start_index: int = 0) -> Dict[str, str]:
    """
    Convert a single CSV file to keyframe mapping format
    
    Args:
        csv_file_path: Path to CSV file
        global_start_index: Starting index for global numbering
    
    Returns:
        dict: Mapping in format {"global_index": "group/video/keyframe"}
    """
    mapping = {}
    
    # Parse video key from filename
    filename = os.path.basename(csv_file_path)
    group_num, video_num = parse_video_key_from_filename(filename)
    
    with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            # Use 'n' column as keyframe number (1-based)
            keyframe_num = int(row['n'])
            
            # Create global index (0-based)
            global_index = global_start_index + keyframe_num - 1
            
            # Create mapping in format "group/video/keyframe"
            mapping[str(global_index)] = f"{group_num}/{video_num}/{keyframe_num}"
    
    return mapping


def process_csv_directory(csv_dir: str, output_file: str = None) -> Dict[str, str]:
    """
    Process all CSV files in a directory and create combined mapping
    
    Args:
        csv_dir: Directory containing CSV files
        output_file: Optional output file path for the combined mapping
    
    Returns:
        dict: Combined mapping from all CSV files
    """
    csv_path = Path(csv_dir)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Directory not found: {csv_dir}")
    
    # Find all CSV files
    csv_files = list(csv_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {csv_dir}")
    
    # Sort files for consistent ordering
    csv_files.sort()
    
    combined_mapping = {}
    global_index = 0
    
    print("Processing CSV files...")
    for csv_file in tqdm(csv_files, desc="Converting CSV files"):
        try:
            # Get mapping for this file
            file_mapping = csv_to_keyframe_mapping(str(csv_file), global_index)
            
            # Update global index for next file
            global_index += len(file_mapping)
            
            # Add to combined mapping
            combined_mapping.update(file_mapping)
            
            print(f"  {csv_file.name}: {len(file_mapping)} keyframes")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    print(f"Total keyframes: {len(combined_mapping)}")
    
    # Save combined mapping
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_mapping, f, indent=2, ensure_ascii=False)
        print(f"Saved mapping to: {output_file}")
    
    return combined_mapping


def create_video_metadata_from_csv(csv_dir: str, output_file: str = None) -> Dict[str, dict]:
    """
    Create video metadata from CSV files
    
    Args:
        csv_dir: Directory containing CSV files
        output_file: Optional output file path
    
    Returns:
        dict: Video metadata
    """
    csv_path = Path(csv_dir)
    csv_files = list(csv_path.glob("*.csv"))
    
    video_metadata = {}
    
    for csv_file in tqdm(csv_files, desc="Creating video metadata"):
        try:
            filename = csv_file.name
            group_num, video_num = parse_video_key_from_filename(filename)
            video_key = f"L{group_num:02d}_V{video_num:03d}"
            
            # Count keyframes in this video
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                keyframe_count = sum(1 for _ in reader)
            
            video_metadata[video_key] = {
                'group_num': group_num,
                'video_num': video_num,
                'keyframe_count': keyframe_count,
                'csv_file': filename
            }
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(video_metadata, f, indent=2, ensure_ascii=False)
        print(f"Saved video metadata to: {output_file}")
    
    return video_metadata


def migrate_csv_to_mongodb(csv_dir: str, mapping_file: str = None):
    """
    Complete migration: Convert CSV files to mapping and migrate to MongoDB
    
    Args:
        csv_dir: Directory containing CSV files
        mapping_file: Optional path to save the mapping file
    """
    print("=== CSV to MongoDB Migration ===")
    
    # Create mapping from CSV files
    if mapping_file is None:
        mapping_file = "resources/embeddings_keys/csv_keyframe_mapping.json"
    
    print(f"Converting CSV files from: {csv_dir}")
    mapping = process_csv_directory(csv_dir, mapping_file)
    
    # Migrate to MongoDB using existing keyframe_migration
    print(f"\nMigrating to MongoDB using mapping: {mapping_file}")
    import asyncio
    asyncio.run(migrate_keyframes(mapping_file))
    
    print("Migration completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Convert CSV keyframe files to keyframe mapping")
    parser.add_argument("--csv_dir", required=True, help="Directory containing CSV files")
    parser.add_argument("--output", help="Output file path for the mapping")
    parser.add_argument("--migrate_mongodb", action="store_true", 
                       help="Also migrate to MongoDB after creating mapping")
    parser.add_argument("--create_metadata", action="store_true",
                       help="Create video metadata file")
    parser.add_argument("--metadata_output", help="Output file for video metadata")
    
    args = parser.parse_args()
    
    try:
        # Create mapping from CSV files
        mapping = process_csv_directory(args.csv_dir, args.output)
        
        # Create video metadata if requested
        if args.create_metadata:
            metadata = create_video_metadata_from_csv(args.csv_dir, args.metadata_output)
            print(f"Created metadata for {len(metadata)} videos")
        
        # Migrate to MongoDB if requested
        if args.migrate_mongodb:
            mapping_file = args.output or "resources/embeddings_keys/csv_keyframe_mapping.json"
            migrate_csv_to_mongodb(args.csv_dir, mapping_file)
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()