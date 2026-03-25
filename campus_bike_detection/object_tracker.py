"""目标追踪模块：BoT-SORT + 遮挡缓冲重匹配"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from campus_bike_detection.models import Detection, Track

_BICYCLE_CLASS_ID = 1
# 自定义 tracker 配置（track_buffer=90，减少遮挡后 ID 切换）
_TRACKER_CFG = str(Path(__file__).parent / "botsort_custom.yaml")

# 缓冲池：轨迹消失后保留的最大帧数，超过则真正删除
_LOST_BUFFER_FRAMES = 60
# 重匹配：新 track 中心点与缓冲池轨迹末尾中心点距离阈值（归一化坐标）
_REMATCH_DIST_THRESH = 0.08


class ObjectTracker:
    """增强追踪器：BoT-SORT + 应用层缓冲重匹配。

    遮挡处理流程：
    1. ultralytics track_buffer=90 先在内部保留 lost 状态
    2. 应用层额外维护 _lost_pool，轨迹消失后再保留 60 帧
    3. 新出现的 track 若中心点与 lost 轨迹末尾距离 < 阈值，复用旧 ID
    """

    def __init__(
        self,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.3,
        device: str = "cpu",
    ) -> None:
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self._device = device
        self._model = None

        # 活跃轨迹池：track_id -> Track
        self._track_pool: dict[int, Track] = {}
        # 丢失缓冲池：track_id -> (Track, lost_frame_count)
        self._lost_pool: dict[int, tuple[Track, int]] = {}
        # 新旧 ID 映射：ultralytics 新 ID -> 复用的旧 ID
        self._id_remap: dict[int, int] = {}
        # 整个会话出现过的唯一 ID（去重后）
        self._all_seen_ids: set[int] = set()

    def set_model(self, model) -> None:
        self._model = model

    def track(self, frame: np.ndarray, frame_id: int, imgsz: int = 640) -> tuple[list[Detection], list[Track]]:
        if self._model is None:
            return [], []

        results = self._model.track(
            frame,
            persist=True,
            classes=[_BICYCLE_CLASS_ID],
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
            tracker=_TRACKER_CFG,
            imgsz=imgsz,
            device=self._device,
        )

        detections: list[Detection] = []
        raw_tracks: list[tuple[int, tuple, float]] = []  # (raw_id, bbox, conf)

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls = int(box.cls[0].item())
                if cls != _BICYCLE_CLASS_ID:
                    continue
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxyn[0].tolist()]
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2), confidence=conf,
                    class_id=cls, frame_id=frame_id,
                ))
                if box.id is not None:
                    raw_id = int(box.id[0].item())
                    raw_tracks.append((raw_id, (x1, y1, x2, y2), conf))

        # ------------------------------------------------------------------
        # 应用层重匹配：新 ID 尝试与 lost_pool 里的旧轨迹匹配
        # ------------------------------------------------------------------
        seen_canonical_ids: set[int] = set()
        active_tracks: list[Track] = []

        for raw_id, bbox, conf in raw_tracks:
            canonical_id = self._id_remap.get(raw_id, raw_id)

            # 如果是全新 ID（不在 track_pool 也不在 remap），尝试与 lost_pool 匹配
            if canonical_id not in self._track_pool and raw_id not in self._id_remap:
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                best_id, best_dist = None, _REMATCH_DIST_THRESH
                for lost_id, (lost_track, _) in self._lost_pool.items():
                    if lost_track.trajectory:
                        lx, ly = lost_track.trajectory[-1]
                        dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_id = lost_id
                if best_id is not None:
                    # 复用旧 ID
                    canonical_id = best_id
                    self._id_remap[raw_id] = canonical_id
                    # 从 lost_pool 恢复到 track_pool
                    self._track_pool[canonical_id] = self._lost_pool.pop(canonical_id)[0]

            seen_canonical_ids.add(canonical_id)
            self._all_seen_ids.add(canonical_id)
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            if canonical_id in self._track_pool:
                t = self._track_pool[canonical_id]
                t.bbox = bbox
                t.age += 1
                t.state = "active"
                t.confidence = conf
                t.trajectory.append((cx, cy))
                if len(t.trajectory) > 60:
                    t.trajectory = t.trajectory[-60:]
            else:
                t = Track(
                    track_id=canonical_id,
                    bbox=bbox,
                    state="active",
                    age=1,
                    confidence=conf,
                    trajectory=[(cx, cy)],
                )
                self._track_pool[canonical_id] = t

            active_tracks.append(t)

        # ------------------------------------------------------------------
        # 将本帧消失的活跃轨迹移入 lost_pool
        # ------------------------------------------------------------------
        lost_ids = set(self._track_pool.keys()) - seen_canonical_ids
        for tid in lost_ids:
            self._lost_pool[tid] = (self._track_pool.pop(tid), 0)

        # lost_pool 计数递增，超过缓冲帧数则真正删除
        expired = [tid for tid, (_, cnt) in self._lost_pool.items()
                   if cnt + 1 > _LOST_BUFFER_FRAMES]
        for tid in expired:
            del self._lost_pool[tid]
        for tid in list(self._lost_pool.keys()):
            if tid not in expired:
                t, cnt = self._lost_pool[tid]
                self._lost_pool[tid] = (t, cnt + 1)

        return detections, active_tracks

    def get_unique_count(self) -> int:
        """当前帧活跃的唯一自行车数。"""
        return len(self._track_pool)

    def get_total_unique_count(self) -> int:
        """整个会话出现过的唯一自行车总数（track ID 去重）。"""
        return len(self._all_seen_ids)
