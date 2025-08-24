import sys
import os
import asyncio
from pathlib import Path
from PIL import Image
import imagehash
from tqdm import tqdm

# Add parent directory to path
ROOT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT_FOLDER)

from app.models.keyframe import Keyframe
from migration.keyframe_migration import init_db
from app.core.settings import AppSettings

async def migrate_phashes():
    await init_db()
    settings = AppSettings()
    data_folder = Path(settings.DATA_FOLDER)
    
    print("Fetching all keyframes from MongoDB...")
    all_keyframes = await Keyframe.find_all().to_list()
    
    print(f"Found {len(all_keyframes)} keyframes. Calculating and updating pHashes...")
    
    updates = []
    for kf in tqdm(all_keyframes, desc="Calculating pHashes"):
        try:
            # Construct image path (adjust if your structure is different)
            img_path = data_folder / f"L{kf.group_num:02d}" / f"L{kf.group_num:02d}_V{kf.video_num:03d}" / f"{kf.keyframe_num:03d}.jpg"
            if not img_path.exists():
                # Try other extensions if needed, e.g., .webp
                continue

            image = Image.open(img_path)
            phash_value = str(imagehash.dhash(image))
            
            # Use update_one to avoid replacing the whole document
            updates.append(Keyframe.find_one(Keyframe.key == kf.key).update({"$set": {Keyframe.phash: phash_value}}))
            
            # Perform updates in batches to avoid overwhelming the server
            if len(updates) >= 500:
                await asyncio.gather(*updates)
                updates = []

        except Exception as e:
            print(f"Could not process keyframe {kf.key}: {e}")

    # Process any remaining updates
    if updates:
        await asyncio.gather(*updates)
        
    print("Finished pHash migration.")

if __name__ == "__main__":
    asyncio.run(migrate_phashes())
