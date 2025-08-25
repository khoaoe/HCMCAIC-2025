import re
from typing import List, Tuple
from collections import OrderedDict

from schema.response import KeyframeServiceReponse


def safe_convert_video_num(video_num: object) -> int:
    """Safely convert video_num to int, handling formats like '26_V288'."""
    if isinstance(video_num, str):
        numeric_part = ''.join(filter(str.isdigit, video_num.split('_')[-1]))
        return int(numeric_part) if numeric_part else 0
    return int(video_num)


def parse_video_id(video_id_str: str) -> Tuple[int, int]:
    """Parse 'Lxx/Lxx_Vxxx' or 'Lxx_Vxxx' into (group_num, video_num)."""
    normalized_id = video_id_str.replace('\\', '/').strip('/')
    numbers = [int(n) for n in re.findall(r'\d+', normalized_id)]
    if len(numbers) >= 2:
        return numbers[0], numbers[-1]
    raise ValueError(f"Could not parse group and video number from '{video_id_str}'")


def deduplicate_and_fuse_keyframes(
    keyframes: List[KeyframeServiceReponse],
) -> List[KeyframeServiceReponse]:
    """
    Deduplicate keyframes by unique key (group, video, frame), keeping the
    highest-confidence instance. Maintains stable order by score.
    """
    if not keyframes:
        return []

    best_keyframes: OrderedDict[tuple[int, int, int], KeyframeServiceReponse] = OrderedDict()

    for kf in sorted(keyframes, key=lambda x: x.confidence_score, reverse=True):
        key = (kf.group_num, kf.video_num, kf.keyframe_num)
        if key not in best_keyframes:
            best_keyframes[key] = kf

    return list(best_keyframes.values())


