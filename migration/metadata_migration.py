import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple
import logging
import sys

# Ensure project root is on sys.path for absolute imports
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import ffmpeg-python for video info extraction
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    print("Warning: ffmpeg-python not available. Video technical info will use default values.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import after setting up logging
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from app.models.video_metadata import VideoMetadata


def _extract_db_name_from_uri(uri: str) -> Optional[str]:
    """Extract database name from Mongo URI if present."""
    try:
        # mongodb://host:port[/db][?params]
        after_slashes = uri.split('://', 1)[-1]
        path = after_slashes.split('/', 1)[1] if '/' in after_slashes else ''
        db_and_params = path.split('?', 1)[0]
        db_name = db_and_params if db_and_params else None
        return db_name
    except Exception:
        return None


class MetadataMigration:
    """Migration script for video metadata"""
    
    def __init__(self, mongo_uri: str, metadata_dir: str, video_dir: Optional[str] = None, mongo_db_name: Optional[str] = None):
        """
        Initialize the migration
        
        Args:
            mongo_uri: MongoDB connection URI
            metadata_dir: Directory containing JSON metadata files
            video_dir: Directory containing video files (optional, for extracting technical info)
            mongo_db_name: Database name (optional). If not provided, will try to parse from URI.
        """
        self.mongo_uri = mongo_uri
        self.metadata_dir = Path(metadata_dir)
        self.video_dir = Path(video_dir) if video_dir else None
        self.client = None
        self.db = None
        self.mongo_db_name = mongo_db_name or _extract_db_name_from_uri(mongo_uri) or os.getenv("MONGO_DB_NAME")
        if not self.mongo_db_name:
            # Final fallback
            self.mongo_db_name = "hcmcaic"
            logger.warning("No database specified; defaulting to 'hcmcaic'. Set MONGO_DB_NAME or include DB in URI to override.")
        
    async def connect_to_mongodb(self):
        """Connect to MongoDB and initialize Beanie"""
        try:
            self.client = AsyncIOMotorClient(self.mongo_uri)
            self.db = self.client.get_database(self.mongo_db_name)
            
            # Initialize Beanie
            await init_beanie(
                database=self.db,
                document_models=[VideoMetadata]
            )
            
            logger.info(f"Successfully connected to MongoDB database '{self.mongo_db_name}'")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def clear_existing_data(self):
        """Clear existing data in video_metadata collection"""
        try:
            await VideoMetadata.delete_all()
            logger.info("Cleared existing video_metadata collection")
        except Exception as e:
            logger.error(f"Failed to clear existing data: {e}")
            raise
    
    def extract_video_info_from_filename(self, filename: str) -> Tuple[str, int, int]:
        """
        Extract video_id, group_num, and video_num from filename
        
        Args:
            filename: JSON filename (e.g., "L30_V001.json")
            
        Returns:
            Tuple of (video_id, group_num, video_num)
        """
        # Extract video_id from filename (remove .json extension)
        video_id = filename.replace('.json', '')
        
        # Parse group_num and video_num from video_id
        # Format: L{group_num}_V{video_num}
        match = re.match(r'L(\d+)_V(\d+)', video_id)
        if match:
            group_num = int(match.group(1))
            video_num = int(match.group(2))
        else:
            # Fallback: try to extract numbers
            numbers = re.findall(r'\d+', video_id)
            if len(numbers) >= 2:
                group_num = int(numbers[0])
                video_num = int(numbers[1])
            else:
                group_num = 0
                video_num = 0
                logger.warning(f"Could not parse group_num and video_num from {video_id}")
        
        return video_id, group_num, video_num
    
    def get_video_info_from_ffprobe(self, video_path: Path) -> Tuple[float, float, int]:
        """
        Get video technical information using ffprobe
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (duration, fps, total_frames)
        """
        if not FFMPEG_AVAILABLE:
            logger.warning("ffmpeg-python not available, using default values")
            return 120.0, 30.0, 3600
        
        try:
            if not video_path.exists():
                logger.warning(f"Video file not found: {video_path}")
                return 120.0, 30.0, 3600
            
            probe = ffmpeg.probe(str(video_path))
            video_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), 
                None
            )
            
            if video_stream is None:
                logger.warning(f"No video stream found in {video_path}")
                return 120.0, 30.0, 3600
            
            # Extract duration
            duration = float(video_stream.get('duration', 120.0))
            
            # Extract fps
            fps_str = video_stream.get('r_frame_rate', '30/1')
            if '/' in fps_str:
                num, den = map(int, fps_str.split('/'))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(fps_str)
            
            # Extract total frames
            total_frames = int(video_stream.get('nb_frames', duration * fps))
            
            return duration, fps, total_frames
            
        except Exception as e:
            logger.warning(f"Failed to get video info for {video_path}: {e}")
            return 120.0, 30.0, 3600
    
    def find_video_file(self, video_id: str) -> Optional[Path]:
        """
        Find corresponding video file for a given video_id
        
        Args:
            video_id: Video ID to search for
            
        Returns:
            Path to video file if found, None otherwise
        """
        if not self.video_dir or not self.video_dir.exists():
            return None
        
        # Common video extensions
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        # Search for video file with matching video_id
        for ext in video_extensions:
            video_path = self.video_dir / f"{video_id}{ext}"
            if video_path.exists():
                return video_path
        
        # If not found, search in subdirectories
        for subdir in self.video_dir.iterdir():
            if subdir.is_dir():
                for ext in video_extensions:
                    video_path = subdir / f"{video_id}{ext}"
                    if video_path.exists():
                        return video_path
        
        return None
    
    async def process_metadata_file(self, json_file: Path) -> Optional[VideoMetadata]:
        """
        Process a single metadata JSON file
        
        Args:
            json_file: Path to JSON metadata file
            
        Returns:
            VideoMetadata object if successful, None otherwise
        """
        try:
            # Read JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Extract video info from filename
            video_id, group_num, video_num = self.extract_video_info_from_filename(json_file.name)
            
            # Extract metadata from JSON
            title = metadata.get('title', '')
            description = metadata.get('description', '')
            keywords = metadata.get('keywords', [])
            author = metadata.get('author', '')
            publish_date = metadata.get('publish_date', '')
            
            # Get video technical info
            if self.video_dir:
                video_path = self.find_video_file(video_id)
                if video_path:
                    duration, fps, total_frames = self.get_video_info_from_ffprobe(video_path)
                else:
                    logger.warning(f"Video file not found for {video_id}")
                    duration, fps, total_frames = 120.0, 30.0, 3600
            else:
                # Use default values if video directory not provided
                duration, fps, total_frames = 120.0, 30.0, 3600
            
            # Create VideoMetadata object
            video_metadata = VideoMetadata(
                video_id=video_id,
                group_num=group_num,
                video_num=video_num,
                title=title,
                description=description,
                keywords=keywords,
                author=author,
                publish_date=publish_date,
                duration=duration,
                total_frames=total_frames,
                fps=fps
            )
            
            return video_metadata
            
        except Exception as e:
            logger.error(f"Failed to process {json_file}: {e}")
            return None
    
    async def migrate_metadata(self):
        """Main migration function"""
        try:
            # Connect to MongoDB
            await self.connect_to_mongodb()
            
            # Clear existing data
            await self.clear_existing_data()
            
            # Find all JSON files
            json_files = list(self.metadata_dir.glob("*.json"))
            logger.info(f"Found {len(json_files)} JSON files to process")
            
            # Process each file
            successful_count = 0
            failed_count = 0
            
            for json_file in json_files:
                logger.info(f"Processing {json_file.name}")
                
                video_metadata = await self.process_metadata_file(json_file)
                if video_metadata:
                    try:
                        # Insert into MongoDB
                        await video_metadata.insert()
                        successful_count += 1
                        logger.info(f"Successfully migrated {video_metadata.video_id}")
                    except Exception as e:
                        logger.error(f"Failed to insert {json_file.name}: {e}")
                        failed_count += 1
                else:
                    failed_count += 1
            
            logger.info(f"Migration completed. Successful: {successful_count}, Failed: {failed_count}")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
        finally:
            if self.client:
                self.client.close()


async def main():
    """Main function to run the migration"""
    # Configuration
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "hcmcaic")
    METADATA_DIR = os.getenv("RESOURCES_METADATA_DIR", "resources/metadata")
    VIDEO_DIR = os.getenv("VIDEO_ROOT")  # Path to video files (optional)
    
    # Create migration instance
    migration = MetadataMigration(
        mongo_uri=MONGO_URI,
        metadata_dir=METADATA_DIR,
        video_dir=VIDEO_DIR,
        mongo_db_name=MONGO_DB_NAME
    )
    
    # Run migration
    await migration.migrate_metadata()


if __name__ == "__main__":
    asyncio.run(main())
