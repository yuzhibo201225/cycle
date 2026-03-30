from __future__ import annotations

from dataclasses import dataclass

from campus_bike_detection.models import Detection, Track


@dataclass(slots=True)
class _State:
    bbox: tuple[float, float, float, float]
    misses: int
    traj: list[tuple[float, float]]


class BikeTracker:
    """Fast IoU tracker with motion/scale gates and lightweight re-id for stable totals."""

    def __init__(
        self,
        iou_thresh: float = 0.3,
        max_misses: int = 20,
        max_center_step: float = 0.18,
        max_area_ratio: float = 2.8,
        reid_window_frames: int = 120,
        reid_max_center_step: float = 0.22,
    ) -> None:
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.max_center_step = max_center_step
        self.max_area_ratio = max_area_ratio
        self.reid_window_frames = reid_window_frames
        self.reid_max_center_step = reid_max_center_step

        self.next_id = 1
        self.states: dict[int, _State] = {}

        self.frame_idx = 0
        self.track_to_global: dict[int, int] = {}
        self.next_global_id = 1
        self.seen_globals: set[int] = set()
        self.recent_ended: list[tuple[int, tuple[float, float, float, float], int]] = []

    def update(self, detections: list[Detection]) -> list[Track]:
        dets = list(detections)
        assigned: dict[int, Detection] = {}

        for tid, state in list(self.states.items()):
            best_det = None
            best_iou = self.iou_thresh
            for det in dets:
                if not self._is_plausible_match(state.bbox, det.bbox, self.max_center_step):
                    continue
                score = _iou(state.bbox, det.bbox)
                if score > best_iou:
                    best_iou = score
                    best_det = det
            if best_det is None:
                state.misses += 1
                if state.misses > self.max_misses:
                    gid = self.track_to_global.pop(tid, None)
                    if gid is not None:
                        self.recent_ended.append((self.frame_idx, state.bbox, gid))
                    self.states.pop(tid, None)
                continue

            dets.remove(best_det)
            cx, cy = _center(best_det.bbox)
            state.bbox = best_det.bbox
            state.misses = 0
            state.traj.append((cx, cy))
            if len(state.traj) > 40:
                state.traj = state.traj[-40:]
            assigned[tid] = best_det

        self._prune_recent_ended()

        for det in dets:
            tid = self.next_id
            self.next_id += 1
            cx, cy = _center(det.bbox)
            self.states[tid] = _State(det.bbox, 0, [(cx, cy)])
            assigned[tid] = det

            gid = self._match_recent_global(det.bbox)
            if gid is None:
                gid = self.next_global_id
                self.next_global_id += 1
            self.track_to_global[tid] = gid
            self.seen_globals.add(gid)

        tracks: list[Track] = []
        for tid, det in assigned.items():
            tracks.append(Track(track_id=tid, bbox=det.bbox, confidence=det.confidence, trajectory=self.states[tid].traj))

        self.frame_idx += 1
        return tracks

    def _is_plausible_match(
        self,
        prev: tuple[float, float, float, float],
        cur: tuple[float, float, float, float],
        max_center_step: float,
    ) -> bool:
        pcx, pcy = _center(prev)
        ccx, ccy = _center(cur)
        if ((pcx - ccx) ** 2 + (pcy - ccy) ** 2) ** 0.5 > max_center_step:
            return False

        pa = max((prev[2] - prev[0]) * (prev[3] - prev[1]), 1e-9)
        ca = max((cur[2] - cur[0]) * (cur[3] - cur[1]), 1e-9)
        ratio = max(pa / ca, ca / pa)
        return ratio <= self.max_area_ratio

    def _prune_recent_ended(self) -> None:
        self.recent_ended = [
            item for item in self.recent_ended if self.frame_idx - item[0] <= self.reid_window_frames
        ]

    def _match_recent_global(self, bbox: tuple[float, float, float, float]) -> int | None:
        best_gid = None
        best_dist = self.reid_max_center_step
        cx, cy = _center(bbox)

        for _, old_bbox, gid in self.recent_ended:
            if not self._is_plausible_match(old_bbox, bbox, self.reid_max_center_step):
                continue
            ox, oy = _center(old_bbox)
            dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_gid = gid

        return best_gid

    def total_unique(self) -> int:
        return len(self.seen_globals)


def _center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    return (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
