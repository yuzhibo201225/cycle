from __future__ import annotations

from campus_bike_detection.models import CountLine, Track


class FlowCounter:
    """Direction-aware counting with jitter and debounce guards."""

    def __init__(self, line: CountLine, direction: str = "both", min_cross: float = 0.003, debounce_frames: int = 5) -> None:
        self.line = line
        self.direction = direction
        self.min_cross = min_cross
        self.debounce_frames = debounce_frames

        self.counted_ids: set[int] = set()
        self.last_side: dict[int, float] = {}
        self.last_count_frame: dict[int, int] = {}
        self.total = 0
        self.forward = 0
        self.backward = 0

    def _point_side(self, p: tuple[float, float]) -> float:
        x1, y1 = self.line.start
        x2, y2 = self.line.end
        px, py = p
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def _is_allowed_direction(self, prev: float, cur: float) -> bool:
        if self.direction == "both":
            return True
        if self.direction == "forward":
            return prev < 0 < cur
        if self.direction == "backward":
            return prev > 0 > cur
        return False

    def update(self, tracks: list[Track], frame_idx: int) -> int:
        for track in tracks:
            cx = (track.bbox[0] + track.bbox[2]) * 0.5
            cy = (track.bbox[1] + track.bbox[3]) * 0.5
            cur_side = self._point_side((cx, cy))

            prev_side = self.last_side.get(track.track_id)
            self.last_side[track.track_id] = cur_side
            if prev_side is None:
                continue

            if track.track_id in self.counted_ids:
                continue

            if prev_side * cur_side >= 0:
                continue

            if not self._is_allowed_direction(prev_side, cur_side):
                continue

            if min(abs(prev_side), abs(cur_side)) < self.min_cross:
                continue

            last_count = self.last_count_frame.get(track.track_id, -10**9)
            if frame_idx - last_count <= self.debounce_frames:
                continue

            self.last_count_frame[track.track_id] = frame_idx
            self.counted_ids.add(track.track_id)
            self.total += 1
            if prev_side < 0 < cur_side:
                self.forward += 1
            elif prev_side > 0 > cur_side:
                self.backward += 1

        return self.total

    def snapshot_counts(self) -> dict[str, int]:
        return {
            self.line.line_id: self.total,
            f"{self.line.line_id}_forward": self.forward,
            f"{self.line.line_id}_backward": self.backward,
        }
