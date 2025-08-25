#!/usr/bin/env python3
"""
NPY Embedding Migration Script for HCMCAIC-2025 (Basic Version)

This script migrates .npy embedding files to Milvus with temporal metadata support.
It can handle individual .npy files or batch process all files in a directory.

Features:
- Loads individual .npy files or batch processes all files
- Creates proper temporal metadata from filename patterns
- Supports filtering by specific groups (e.g., L21-L30 only)
- Generates id2index mapping for temporal search
- Migrates to Milvus with proper indexing
"""

import os
import numpy as np
import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Try to import pymilvus, but handle gracefully if not available
try:
    from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    print("Warning: pymilvus not available. Migration to Milvus will be skipped.")
    MILVUS_AVAILABLE = False


class NPYEmbeddingMigrator:
    def __init__(self, embeddings_dir: str = "resources/embeddings"):
        """
        Initialize the NPY embedding migrator
        
        Args:
            embeddings_dir: Directory containing .npy files
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.id2index_mapping = {}
        self.current_index = 0
        
    def parse_filename(self, filename: str) -> Tuple[int, int, int]:
        """
        Parse filename like 'L21_V015.npy' to extract group, video, and keyframe info
        
        Args:
            filename: Filename to parse
            
        Returns:
            Tuple of (group_num, video_num, keyframe_count)
        """
        # Extract group number (L21 -> 21)
        group_match = re.search(r'L(\d+)', filename)
        if not group_match:
            raise ValueError(f"Invalid filename format: {filename}")
        
        group_num = int(group_match.group(1))
        
        # Extract video number (V015 -> 15)
        video_match = re.search(r'V(\d+)', filename)
        if not video_match:
            raise ValueError(f"Invalid filename format: {filename}")
        
        video_num = int(video_match.group(1))
        
        # For .npy files, each file contains all keyframes for that video
        # We'll need to load the file to get the actual keyframe count
        return group_num, video_num, 0  # Will be updated when loading
        
    def load_npy_file(self, filepath: Path) -> np.ndarray:
        """
        Load a .npy file and return the embeddings array with consistent data type
        
        Args:
            filepath: Path to the .npy file
            
        Returns:
            NumPy array of embeddings with consistent dtype (float32)
        """
        try:
            embeddings = np.load(filepath)
            
            # Check and convert data type for consistency
            if embeddings.dtype != np.float32:
                print(f"[INFO] Converting {filepath.name} from {embeddings.dtype} to float32")
                embeddings = embeddings.astype(np.float32)
            
            # Only print for first few files to avoid spam
            if hasattr(self, '_files_loaded'):
                self._files_loaded += 1
            else:
                self._files_loaded = 1
                print(f"[FOLDER] Loading files (showing first 10):")
            
            if self._files_loaded <= 10:
                print(f"   {self._files_loaded:2d}. {filepath.name}: shape {embeddings.shape}, dtype {embeddings.dtype}")
            elif self._files_loaded == 11:
                print(f"   ... (continuing silently)")
            
            return embeddings
        except Exception as e:
            print(f"[ERROR] Error loading {filepath}: {e}")
            return None
    
    def get_available_groups(self) -> List[str]:
        """
        Get list of available groups from .npy files
        
        Returns:
            List of group names (e.g., ['L01', 'L02', ..., 'L30'])
        """
        groups = set()
        for filepath in self.embeddings_dir.glob("*.npy"):
            group_match = re.search(r'L(\d+)', filepath.name)
            if group_match:
                groups.add(f"L{group_match.group(1)}")
        return sorted(list(groups))
    
    def filter_files_by_groups(self, groups: Optional[List[str]] = None) -> List[Path]:
        """
        Filter .npy files by specified groups
        
        Args:
            groups: List of groups to include (e.g., ['L21', 'L22', ..., 'L30'])
                   If None, includes all groups
                   
        Returns:
            List of file paths matching the filter
        """
        if groups is None:
            return list(self.embeddings_dir.glob("*.npy"))
        
        filtered_files = []
        for group in groups:
            pattern = f"{group}_V*.npy"
            filtered_files.extend(self.embeddings_dir.glob(pattern))
        
        return sorted(filtered_files)
    
    def create_id2index_mapping(self, files: List[Path]) -> Dict[str, str]:
        """
        Create id2index mapping for temporal metadata
        
        Args:
            files: List of .npy file paths
            
        Returns:
            Dictionary mapping index to "group/video/keyframe" format
        """
        id2index = {}
        current_index = 0
        
        for filepath in tqdm(files, desc="Creating id2index mapping"):
            try:
                group_num, video_num, _ = self.parse_filename(filepath.name)
                embeddings = self.load_npy_file(filepath)
                
                if embeddings is not None:
                    num_keyframes = embeddings.shape[0]
                    
                    # Create mapping for each keyframe in this video
                    for keyframe_idx in range(num_keyframes):
                        id2index[str(current_index)] = f"{group_num}/{video_num}/{keyframe_idx + 1}"
                        current_index += 1
                        
            except Exception as e:
                print(f"Error processing {filepath.name}: {e}")
                continue
        
        return id2index
    
    def combine_embeddings(self, files: List[Path]) -> np.ndarray:
        """
        Combine all embeddings from .npy files into a single array
        
        Args:
            files: List of .npy file paths
            
        Returns:
            Combined NumPy array of all embeddings
        """
        all_embeddings = []
        embedding_dim = None
        
        for filepath in tqdm(files, desc="Loading embeddings"):
            try:
                embeddings = self.load_npy_file(filepath)
                if embeddings is not None:
                    # Check embedding dimension consistency
                    if embedding_dim is None:
                        embedding_dim = embeddings.shape[1]
                    elif embeddings.shape[1] != embedding_dim:
                        print(f"[WARNING]  Warning: {filepath.name} has dimension {embeddings.shape[1]}, expected {embedding_dim}")
                        print(f"   Skipping this file to avoid dimension mismatch")
                        continue
                    
                    all_embeddings.append(embeddings)
            except Exception as e:
                print(f"Error loading {filepath.name}: {e}")
                continue
        
        if not all_embeddings:
            raise ValueError("No valid embeddings found")
        
        # Concatenate all embeddings
        combined = np.vstack(all_embeddings)
        print(f"Combined embeddings shape: {combined.shape}")
        return combined
    
    def save_id2index_mapping(self, id2index: Dict[str, str], output_path: str):
        """
        Save id2index mapping to JSON file
        
        Args:
            id2index: Mapping dictionary
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            json.dump(id2index, f, indent=2)
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"[SAVE] Saved id2index mapping to {output_path} ({file_size:.2f} MB)")
    
    def save_combined_embeddings(self, embeddings: np.ndarray, output_path: str):
        """
        Save combined embeddings as .npy file
        
        Args:
            embeddings: Combined embeddings array
            output_path: Output file path
        """
        np.save(output_path, embeddings)
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"[SAVE] Saved combined embeddings to {output_path} ({file_size:.2f} MB)")
    
    def analyze_embeddings(self, files: List[Path]):
        """
        Analyze .npy files and create mapping without migrating to Milvus
        
        Args:
            files: List of .npy file paths to analyze
        """
        print(f"[SEARCH] Analyzing {len(files)} .npy files...")
        print(f"[FILE] Files to process:")
        for i, file in enumerate(files[:5]):  # Show first 5 files
            print(f"   {i+1}. {file.name}")
        if len(files) > 5:
            print(f"   ... and {len(files) - 5} more files")
        print("-" * 60)
        
        # Create id2index mapping
        print("🗺️  Creating temporal metadata mapping...")
        id2index = self.create_id2index_mapping(files)
        
        if not id2index:
            raise ValueError("No valid mappings created")
        
        # Combine embeddings
        print("[LINK] Combining embeddings...")
        combined_embeddings = self.combine_embeddings(files)
        
        num_vectors, embedding_dim = combined_embeddings.shape
        
        print(f"\n[STATS] Analysis Results:")
        print(f"   Total files processed: {len(files)}")
        print(f"   Total embeddings: {num_vectors:,}")
        print(f"   Embedding dimension: {embedding_dim}")
        print(f"   Mapping entries: {len(id2index):,}")
        
        # Group statistics
        group_stats = {}
        for idx_str, mapping in id2index.items():
            group_num = mapping.split('/')[0]
            if group_num not in group_stats:
                group_stats[group_num] = 0
            group_stats[group_num] += 1
        
        print(f"\n[STATS] Group Statistics:")
        for group in sorted(group_stats.keys(), key=int):
            print(f"   L{group}: {group_stats[group]:,} embeddings")
        
        return id2index, combined_embeddings
    
    def migrate_to_milvus(
        self,
        files: List[Path],
        output_mapping_path: str = "resources/keyframes/id2index.json",
        output_embeddings_path: str = "resources/embeddings/combined_embeddings.npy",
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        collection_name: str = "keyframe_embeddings",
        batch_size: int = 10000,
        default_fps: float = 25.0
    ):
        """
        Migrate .npy files to Milvus (if available)
        
        Args:
            files: List of .npy file paths to migrate
            output_mapping_path: Path to save id2index mapping
            output_embeddings_path: Path to save combined embeddings
            milvus_host: Milvus host
            milvus_port: Milvus port
            collection_name: Collection name
            batch_size: Batch size for insertion
            default_fps: Default frames per second for timestamp calculation
        """
        if not MILVUS_AVAILABLE:
            print("[ERROR] Milvus not available. Skipping migration to Milvus.")
            print("   Saving mapping and combined embeddings only.")
            
            # Analyze and save files
            print("[STATS] Step 1/3: Analyzing embeddings...")
            id2index, combined_embeddings = self.analyze_embeddings(files)
            
            # Save mapping
            print("[SAVE] Step 2/3: Saving id2index mapping...")
            os.makedirs(os.path.dirname(output_mapping_path), exist_ok=True)
            self.save_id2index_mapping(id2index, output_mapping_path)
            
            # Save combined embeddings
            print("[SAVE] Step 3/3: Saving combined embeddings...")
            os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)
            self.save_combined_embeddings(combined_embeddings, output_embeddings_path)
            
            print("[OK] Migration completed (Milvus skipped)")
            return
        
        print(f"[START] Starting migration of {len(files)} .npy files to Milvus...")
        print(f"[FILE] Output mapping: {output_mapping_path}")
        print(f"[FILE] Output embeddings: {output_embeddings_path}")
        print(f"[FIX] Batch size: {batch_size:,}")
        print(f"[VIDEO] Default FPS: {default_fps}")
        print("-" * 60)
        
        # Step 1: Analyze embeddings
        print("[STATS] Step 1/5: Analyzing embeddings...")
        id2index, combined_embeddings = self.analyze_embeddings(files)
        # Sanity check to avoid mismatched mapping leading to bad fallbacks
        if len(id2index) != combined_embeddings.shape[0]:
            print(f"[ERROR] Mapping entries ({len(id2index):,}) != embeddings rows ({combined_embeddings.shape[0]:,})")
            raise ValueError("id2index size mismatch with combined embeddings. Ensure inputs exclude combined files and are consistent.")
        
        # Step 2: Save mapping and embeddings
        print("[SAVE] Step 2/5: Saving files locally...")
        os.makedirs(os.path.dirname(output_mapping_path), exist_ok=True)
        self.save_id2index_mapping(id2index, output_mapping_path)
        
        os.makedirs(os.path.dirname(output_embeddings_path), exist_ok=True)
        self.save_combined_embeddings(combined_embeddings, output_embeddings_path)
        
        # Step 3: Connect to Milvus
        print("[CONNECT] Step 3/5: Connecting to Milvus...")
        alias = "default"
        
        if connections.has_connection(alias):
            print("   Removing existing connection...")
            connections.remove_connection(alias)
        
        print(f"   Connecting to {milvus_host}:{milvus_port}...")
        connections.connect(alias=alias, host=milvus_host, port=milvus_port)
        print(f"[OK] Connected to Milvus at {milvus_host}:{milvus_port}")
        
        # Step 4: Create collection and indexes
        print("[BUILD]  Step 4/5: Creating collection and indexes...")
        num_vectors, embedding_dim = combined_embeddings.shape
        print(f"   Embedding dimension: {embedding_dim}")
        print(f"   Total vectors: {num_vectors:,}")
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            # Temporal search fields
            FieldSchema(name="timestamp", dtype=DataType.DOUBLE, description="Timestamp in seconds"),
            FieldSchema(name="group_num", dtype=DataType.INT32, description="Video group number"),
            FieldSchema(name="video_num", dtype=DataType.INT32, description="Video number within group"),
            FieldSchema(name="keyframe_num", dtype=DataType.INT32, description="Keyframe number within video")
        ]
        
        schema = CollectionSchema(fields, f"Collection for {collection_name} embeddings with temporal support")
        
        # Drop existing collection if it exists
        if utility.has_collection(collection_name, using=alias):
            print(f"   Dropping existing collection '{collection_name}'...")
            utility.drop_collection(collection_name, using=alias)
        
        # Create collection
        print(f"   Creating collection '{collection_name}'...")
        collection = Collection(collection_name, schema, using=alias)
        print(f"[OK] Created collection '{collection_name}' with dimension {embedding_dim}")
        
        # Create indexes
        print("   Creating indexes...")
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
        }
        
        collection.create_index("embedding", index_params)
        print("   [OK] Created index for embedding field")
        
        # Create indexes on temporal fields for efficient filtering
        collection.create_index("timestamp", {"index_type": "STL_SORT"})
        collection.create_index("group_num", {"index_type": "STL_SORT"})
        collection.create_index("video_num", {"index_type": "STL_SORT"})
        print("   [OK] Created indexes for temporal fields")
        
        # Step 5: Insert embeddings in batches
        print("[INSERT] Step 5/5: Inserting embeddings into Milvus...")
        total_batches = (num_vectors + batch_size - 1) // batch_size
        print(f"   Total batches: {total_batches}")
        print(f"   Vectors per batch: {batch_size:,}")
        print(f"   Total vectors: {num_vectors:,}")
        print("-" * 60)
        
        for i in tqdm(range(0, num_vectors, batch_size), desc="Inserting batches"):
            end_idx = min(i + batch_size, num_vectors)
            batch_embeddings = combined_embeddings[i:end_idx].tolist()
            batch_ids = list(range(i, end_idx))
            
            # Prepare temporal metadata for each batch item
            batch_timestamps = []
            batch_group_nums = []
            batch_video_nums = []
            batch_keyframe_nums = []
            
            for idx in range(i, end_idx):
                idx_str = str(idx)
                if idx_str not in id2index:
                    raise ValueError(f"Missing id2index entry for id {idx_str}. Mapping and embeddings misaligned.")
                # Parse "group/video/keyframe" format
                parts = id2index[idx_str].split('/')
                if len(parts) < 3:
                    raise ValueError(f"Malformed id2index value for id {idx_str}: '{id2index[idx_str]}'")
                group_num = int(parts[0])
                video_num = int(parts[1])
                keyframe_num = int(parts[2])
                timestamp = keyframe_num / default_fps  # Convert to seconds
                
                batch_timestamps.append(timestamp)
                batch_group_nums.append(group_num)
                batch_video_nums.append(video_num)
                batch_keyframe_nums.append(keyframe_num)
            
            entities = [
                batch_ids, 
                batch_embeddings, 
                batch_timestamps,
                batch_group_nums,
                batch_video_nums,
                batch_keyframe_nums
            ]
            collection.insert(entities)
        
        collection.flush()
        print("[SAVE] Data flushed to disk")
        
        collection.load()
        print("[LOAD] Collection loaded for search")
        
        # Get collection info
        num_entities = collection.num_entities
        print("-" * 60)
        print(f"[SUCCESS] SUCCESS! Migrated {num_entities:,} embeddings to Milvus!")
        print(f"[STATS] Collection: {collection_name}")
        print(f"[LINK] Host: {milvus_host}:{milvus_port}")
        print(f"[FILE] Mapping saved to: {output_mapping_path}")
        print(f"[FILE] Embeddings saved to: {output_embeddings_path}")
        print("-" * 60)
        
        return collection


def main():
    parser = argparse.ArgumentParser(
        description="Migrate .npy embedding files to Milvus",
        epilog="""
Examples:
  # List available groups
  python npy_embedding_migration.py --list_groups
  
  # Migrate all files from a specific folder
  python npy_embedding_migration.py --folder_path resources/embeddings
  
  # Analyze all files from a specific folder
  python npy_embedding_migration.py --folder_path resources/embeddings --analyze_only
  
  # Migrate specific groups
  python npy_embedding_migration.py --groups L21 L22 L23 L24 L25 L26 L27 L28 L29 L30
  
  # Migrate all files from default folder
  python npy_embedding_migration.py
        """
    )
    parser.add_argument(
        "--embeddings_dir", 
        type=str, 
        default="resources/embeddings",
        help="Directory containing .npy files"
    )
    parser.add_argument(
        "--groups", 
        nargs="+", 
        help="Specific groups to migrate (e.g., L21 L22 L23 L24 L25 L26 L27 L28 L29 L30)"
    )
    parser.add_argument(
        "--output_mapping", 
        type=str, 
        default="resources/keyframes/id2index.json",
        help="Output path for id2index mapping"
    )
    parser.add_argument(
        "--output_embeddings", 
        type=str, 
        default="resources/embeddings/combined_embeddings.npy",
        help="Output path for combined embeddings"
    )
    parser.add_argument(
        "--list_groups", 
        action="store_true",
        help="List available groups and exit"
    )
    parser.add_argument(
        "--analyze_only", 
        action="store_true",
        help="Only analyze files, don't migrate to Milvus"
    )
    parser.add_argument(
        "--milvus_host", 
        type=str, 
        default="localhost",
        help="Milvus host"
    )
    parser.add_argument(
        "--milvus_port", 
        type=str, 
        default="19530",
        help="Milvus port"
    )
    parser.add_argument(
        "--collection_name", 
        type=str, 
        default="keyframe_embeddings",
        help="Milvus collection name"
    )
    parser.add_argument(
        "--folder_path", 
        type=str, 
        help="Folder path containing .npy files to migrate (e.g., resources/embeddings)"
    )
    
    args = parser.parse_args()
    
    # Initialize migrator
    migrator = NPYEmbeddingMigrator(args.embeddings_dir)
    
    # List available groups if requested
    if args.list_groups:
        groups = migrator.get_available_groups()
        print("Available groups:")
        for group in groups:
            print(f"  {group}")
        return
    
    # Handle folder path migration
    if args.folder_path:
        folder_path = Path(args.folder_path)
        if not folder_path.exists():
            print(f"[ERROR] Folder not found: {args.folder_path}")
            return
        
        if not folder_path.is_dir():
            print(f"[ERROR] Path is not a directory: {args.folder_path}")
            return
        
        # Get all .npy files from the specified folder
        files = [f for f in folder_path.glob("*.npy") if f.name != "combined_embeddings.npy"]
        
        if not files:
            print(f"[ERROR] No .npy files found in folder: {args.folder_path}")
            return
        
        print(f"Migrating all .npy files from folder: {folder_path}")
    else:
        # Get files to migrate by groups from default embeddings directory
        files = migrator.filter_files_by_groups(args.groups)
        
        if not files:
            print("[ERROR] No .npy files found matching the criteria")
            return
    
    print(f"Found {len(files)} .npy file(s) to migrate")
    
    # Perform migration or analysis
    try:
        if args.analyze_only:
            # Only analyze files
            migrator.analyze_embeddings(files)
        else:
            # Migrate to Milvus (if available)
            migrator.migrate_to_milvus(
                files=files,
                output_mapping_path=args.output_mapping,
                output_embeddings_path=args.output_embeddings,
                milvus_host=args.milvus_host,
                milvus_port=args.milvus_port,
                collection_name=args.collection_name
            )
    except Exception as e:
        print(f"[ERROR] Operation failed: {e}")
        return


if __name__ == "__main__":
    main()
