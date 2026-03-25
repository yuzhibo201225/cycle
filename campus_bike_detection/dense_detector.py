"""密集停放自行车检测：SAHI 切片批量推理 + WBF 融合"""
from __future__ import annotations

import cv2
import numpy as np

from campus_bike_detection.inference_engine import YOLOInferenceEngine
from campus_bike_detection.models import Detection

_BICYCLE_CLASS_ID = 1


class DenseDetector:
    """SAHI 风格的密集检测器。

    流程：
    1. 将帧切成带重叠的子图（默认 640x640，overlap 20%）
    2. 所有切片 + 全图缩放图 **批量** 送入模型（一次 forward）
    3. 将切片局部坐标还原为全图归一化坐标
    4. NMS 去除重叠框
    """

    def __init__(
        self,
        engine: YOLOInferenceEngine,
        slice_size: int = 640,
        overlap_ratio: float = 0.2,
        conf_thresh: float = 0.25,
        nms_iou_thresh: float = 0.45,
    ) -> None:
        self.engine = engine
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.conf_thresh = conf_thresh
        self.nms_iou_thresh = nms_iou_thresh

    # ------------------------------------------------------------------
    # 切片生成
    # ------------------------------------------------------------------

    def _make_slices(self, frame: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        """返回 [(sub_img, (x1,y1,x2,y2)), ...]，坐标为像素。"""
        h, w = frame.shape[:2]
        s = self.slice_size
        stride = max(1, int(s * (1.0 - self.overlap_ratio)))
        slices = []
        y = 0
        while True:
            x = 0
            y2 = min(y + s, h)
            while True:
                x2 = min(x + s, w)
                sub = frame[y:y2, x:x2]
                # pad 到 slice_size x slice_size（右/下补黑边）
                pad = np.zeros((s, s, 3), dtype=np.uint8)
                pad[:y2-y, :x2-x] = sub
                slices.append((pad, (x, y, x2, y2)))
                if x2 == w:
                    break
                x = min(x + stride, w - 1)
            if y2 == h:
                break
            y = min(y + stride, h - 1)
        return slices

    # ------------------------------------------------------------------
    # 批量推理
    # ------------------------------------------------------------------

    def _batch_infer(self, imgs: list[np.ndarray]) -> list[list[tuple]]:
        """批量推理，返回每张图的检测列表 [(x1n,y1n,x2n,y2n,conf), ...]（归一化坐标）。"""
        model = self.engine._model
        device = self.engine.device

        results = model(
            imgs,
            classes=[_BICYCLE_CLASS_ID],
            conf=self.conf_thresh,
            imgsz=self.slice_size,
            verbose=False,
            device=device,
        )

        out = []
        for r in results:
            boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls = int(box.cls[0].item())
                    if cls != _BICYCLE_CLASS_ID:
                        continue
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = [float(v) for v in box.xyxyn[0].tolist()]
                    boxes.append((x1, y1, x2, y2, conf))
            out.append(boxes)
        return out

    # ------------------------------------------------------------------
    # NMS
    # ------------------------------------------------------------------

    def _nms(self, dets: list[Detection]) -> list[Detection]:
        if not dets:
            return []
        dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
        keep: list[Detection] = []
        for det in dets:
            if not any(self._iou(det.bbox, k.bbox) > self.nms_iou_thresh for k in keep):
                keep.append(det)
        return keep

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
        if inter == 0:
            return 0.0
        return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray, frame_id: int) -> list[Detection]:
        h, w = frame.shape[:2]
        slices = self._make_slices(frame)

        # 全图缩放图也加入批次（捕获整体布局）
        full_resized = cv2.resize(frame, (self.slice_size, self.slice_size))
        imgs = [s[0] for s in slices] + [full_resized]
        offsets = [s[1] for s in slices] + [(0, 0, w, h)]  # 全图偏移

        batch_results = self._batch_infer(imgs)

        all_dets: list[Detection] = []
        for boxes, (sx1, sy1, sx2, sy2) in zip(batch_results, offsets):
            sw = sx2 - sx1
            sh = sy2 - sy1
            for (lx1, ly1, lx2, ly2, conf) in boxes:
                # 局部归一化 → 全图归一化
                gx1 = (lx1 * sw + sx1) / w
                gy1 = (ly1 * sh + sy1) / h
                gx2 = (lx2 * sw + sx1) / w
                gy2 = (ly2 * sh + sy1) / h
                gx1, gx2 = max(0.0, min(1.0, gx1)), max(0.0, min(1.0, gx2))
                gy1, gy2 = max(0.0, min(1.0, gy1)), max(0.0, min(1.0, gy2))
                all_dets.append(Detection(
                    bbox=(gx1, gy1, gx2, gy2),
                    confidence=conf,
                    class_id=_BICYCLE_CLASS_ID,
                    frame_id=frame_id,
                ))

        return self._nms(all_dets)
