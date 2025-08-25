import sys
import os
ROOT_FOLDER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)
sys.path.insert(0, ROOT_FOLDER)



from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
import json
import asyncio
import argparse
import time


from app.core.settings import MongoDBSettings
from app.models.keyframe import Keyframe

SETTING = MongoDBSettings()

async def init_db():
    start = time.perf_counter()
    print("[DB] Initializing MongoDB connection...")
    if SETTING.MONGO_URI:
        print("[DB] Using MONGO_URI from environment")
        client = AsyncIOMotorClient(SETTING.MONGO_URI)
    else:
        print(f"[DB] Using host/port auth settings to connect: host={SETTING.MONGO_HOST}, port={SETTING.MONGO_PORT}")
        client = AsyncIOMotorClient(
            host=SETTING.MONGO_HOST,
            port=SETTING.MONGO_PORT,
            username=SETTING.MONGO_USER,
            password=SETTING.MONGO_PASSWORD,
        )
    await init_beanie(database=client[SETTING.MONGO_DB], document_models=[Keyframe])
    dur = (time.perf_counter() - start) * 1000
    print(f"[DB] MongoDB initialized for database='{SETTING.MONGO_DB}' in {dur:.1f} ms")


def load_json_data(file_path):
    print(f"[LOAD] Loading keyframe mapping from: {file_path}")
    t0 = time.perf_counter()
    data = json.load(open(file_path, 'r', encoding='utf-8'))
    dur = (time.perf_counter() - t0) * 1000
    print(f"[LOAD] Loaded JSON with {len(data):,} entries in {dur:.1f} ms")
    return data


def transform_data(data: dict[str,str]) -> list[Keyframe]:
    """
    Convert the data from the old format to the new Keyframe model.
    """
    print("[TRANSFORM] Converting mapping to Keyframe documents...")
    t0 = time.perf_counter()
    keyframes: list[Keyframe] = []
    errors = 0
    shown_samples = 0
    for idx, (key, value) in enumerate(data.items(), start=1):
        try:
            group, video, keyframe = value.split('/')
            keyframe_obj = Keyframe(
                key=int(key),
                video_num=int(video),
                group_num=int(group),
                keyframe_num=int(keyframe)
            )
            keyframes.append(keyframe_obj)
            # Show a few samples early for verification
            if shown_samples < 3:
                print(f"  [SAMPLE] key={key} -> group={group}, video={video}, keyframe={keyframe}")
                shown_samples += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [WARN] Failed to transform entry key='{key}', value='{value}': {e}")
    dur = (time.perf_counter() - t0) * 1000
    print(f"[TRANSFORM] Built {len(keyframes):,} Keyframe docs with {errors} error(s) in {dur:.1f} ms")
    return keyframes

async def migrate_keyframes(file_path):
    print("[MIGRATE] Starting keyframe migration...")
    overall_t0 = time.perf_counter()
    await init_db()
    data = load_json_data(file_path)
    # Quick stats by group for visibility
    try:
        group_counts = {}
        for v in data.values():
            parts = v.split('/')
            if len(parts) >= 1:
                g = parts[0]
                group_counts[g] = group_counts.get(g, 0) + 1
        if group_counts:
            top_groups = ", ".join([f"L{g}:{c:,}" for g, c in sorted(group_counts.items(), key=lambda x: int(x[0]))[:10]])
            print(f"[MIGRATE] Group distribution (first 10 by id): {top_groups}")
    except Exception:
        pass

    keyframes = transform_data(data)

    # Delete existing collection
    t_del = time.perf_counter()
    print("[MIGRATE] Deleting existing keyframes collection (if any)...")
    await Keyframe.delete_all()
    print(f"[MIGRATE] Delete completed in {(time.perf_counter()-t_del)*1000:.1f} ms")

    # Insert new documents
    t_ins = time.perf_counter()
    print(f"[MIGRATE] Inserting {len(keyframes):,} keyframes...")
    await Keyframe.insert_many(keyframes)
    print(f"[MIGRATE] Insert completed in {(time.perf_counter()-t_ins)*1000:.1f} ms")
    print(f"[SUCCESS] Inserted {len(keyframes):,} keyframes into the database.")
    print(f"[DONE] Total migration time: {(time.perf_counter()-overall_t0):.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate keyframes to MongoDB.")
    parser.add_argument(
        "--file_path", type=str, help="Path to the JSON file containing keyframe data."
    )
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"File {args.file_path} does not exist.")
        sys.exit(1)

    asyncio.run(migrate_keyframes(args.file_path))

