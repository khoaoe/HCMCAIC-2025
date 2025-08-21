#!/usr/bin/env python3
"""
Script to run the migration from .pt file to individual .npy files
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
ROOT_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, ROOT_FOLDER)

from migration.npy_migration import migrate_pt_to_npy_files, create_mapping_from_existing_npy_files
from migration.embedding_migration import inject_embeddings_from_npy_simple
from app.core.settings import KeyFrameIndexMilvusSetting


def main():
    # Configuration
    pt_file_path = "resources/embeddings_keys/CLIP_convnext_xxlarge_laion2B_s34B_b82K_augreg_soup_clip_embeddings.pt"
    id2index_path = "resources/embeddings_keys/id2index.json"
    npy_output_dir = "resources/clip-features-32"
    
    print("=== Migration from .pt to .npy files ===")
    print(f"PT file: {pt_file_path}")
    print(f"ID2Index: {id2index_path}")
    print(f"Output directory: {npy_output_dir}")
    
    # Check if files exist
    if not os.path.exists(pt_file_path):
        print(f"Error: PT file not found: {pt_file_path}")
        return
    
    if not os.path.exists(id2index_path):
        print(f"Error: ID2Index file not found: {id2index_path}")
        return
    
    # Check if .npy files already exist
    npy_files = list(Path(npy_output_dir).glob("*.npy"))
    if npy_files:
        print(f"Found {len(npy_files)} existing .npy files in {npy_output_dir}")
        choice = input("Do you want to create mapping for existing files? (y/n): ")
        if choice.lower() == 'y':
            print("Creating mapping for existing .npy files...")
            create_mapping_from_existing_npy_files(npy_output_dir)
            return
    
    # Run migration
    print("\nStarting migration...")
    try:
        migrate_pt_to_npy_files(
            pt_file_path=pt_file_path,
            id2index_path=id2index_path,
            output_dir=npy_output_dir,
            batch_size=1000
        )
        print("Migration completed successfully!")
        
        # Option to inject to Milvus
        choice = input("\nDo you want to inject embeddings to Milvus? (y/n): ")
        if choice.lower() == 'y':
            print("Injecting embeddings to Milvus...")
            setting = KeyFrameIndexMilvusSetting()
            inject_embeddings_from_npy_simple(npy_output_dir, setting)
            print("Milvus injection completed!")
            
    except Exception as e:
        print(f"Error during migration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
