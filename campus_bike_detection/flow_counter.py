from __future__ import annotations

from campus_bike_detection.models import CountLine, Track


class FlowCounter:
    def __init__(self, line: CountLine) -> None:
        self.line = line
        self.counted_ids: set[int] = set()
        self.total = 0

    @staticmethod
    def _ccw(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    def _intersect(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
    ) -> bool:
        return self._ccw(p1, p3, p4) != self._ccw(p2, p3, p4) and self._ccw(p1, p2, p3) != self._ccw(p1, p2, p4)

    def update(self, tracks: list[Track]) -> int:
        for track in tracks:
            if track.track_id in self.counted_ids or len(track.trajectory) < 2:
                continue
            if self._intersect(track.trajectory[-2], track.trajectory[-1], self.line.start, self.line.end):
                self.counted_ids.add(track.track_id)
                self.total += 1
        return self.total
