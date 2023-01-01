from dataclasses import dataclass


@dataclass
class Recommendation:
    song_id: str
    difficulty: int
    characteristic: int
    meta_sim: float
    tag_sim: float
    total_sim: float
