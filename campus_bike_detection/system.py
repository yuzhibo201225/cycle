"""系统主循环与结果输出：BikeDetectionSystem"""
from __future__ import annotations

import time
import uuid
from collections import deque
from typing import Optional

import cv2
import numpy as np

from campus_bike_detection.dense_detector import DenseDetector
from campus_bike_detection.flow_counter import FlowCounter
from campus_bike_detection.frame_capture import FrameCapture
from campus_bike_detection.inference_engine import YOLOInferenceEngine
from campus_bike_detection.models import (
    Detection,
    FlowEvent,
    SessionReport,
    SystemConfig,
    Track,
)
from campus_bike_detection.object_tracker import ObjectTracker


class BikeDetectionSystem:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config

        self.capture = FrameCapture(config.source)
        ok, test_frame = self.capture.read_frame()
        if not ok or test_frame is None:
            raise RuntimeError(f"无法读取输入源: {config.source}")
        print(f"[INFO] 输入源正常，分辨率: {test_frame.shape[1]}x{test_frame.shape[0]}")

        self.engine = YOLOInferenceEngine(config.model_path, config.device)
        self.engine.warmup()

        actual_device = self.engine.device

        # static/dynamic/both 模式都使用追踪器（去重 + 稳定计数）
        self.tracker = ObjectTracker(conf_thresh=0.25, iou_thresh=0.3, device=actual_device)
        self.tracker.set_model(self.engine._model)
        self.flow_counter: Optional[FlowCounter] = None
        if config.mode in ("dynamic", "both"):
            self.flow_counter = FlowCounter(config.count_lines)

        # both 模式额外用 DenseDetector 补充密集区域漏检
        self.dense_detector: Optional[DenseDetector] = None
        if config.mode == "both":
            self.dense_detector = DenseDetector(self.engine)

    def __enter__(self) -> "BikeDetectionSystem":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.capture.release()

    def run(self) -> SessionReport:
        config = self.config
        start_time = time.time()
        session_id = str(uuid.uuid4())

        frame_id = 0
        total_frames = 0
        parked_counts: list[int] = []
        flow_timeline: list[tuple[float, int]] = []
        total_flow_by_line: dict[str, int] = {}
        fps_times: deque[float] = deque(maxlen=30)
        total_cross_count = 0

        while True:
            ok, frame = self.capture.read_frame()
            if not ok or frame is None:
                break
            # 跳帧：尝试拿最新帧，防止视频文件处理时积压
            ok2, frame2 = self.capture.read_frame()
            if ok2 and frame2 is not None:
                frame = frame2

            t0 = time.time()
            timestamp = t0 - start_time

            detections: list[Detection] = []
            tracks: list[Track] = []

            # 所有模式都走追踪器，track ID 去重避免同一辆车重复计数
            detections, tracks = self.tracker.track(frame, frame_id, imgsz=640)
            parked_count = len(tracks)  # 当前帧唯一自行车数

            # both 模式：用 DenseDetector 补充追踪器漏检的密集区域目标
            if self.dense_detector is not None:
                dense_dets = self.dense_detector.detect(frame, frame_id)
                existing = [d.bbox for d in detections]
                for d in dense_dets:
                    if not any(_iou(d.bbox, b) > 0.5 for b in existing):
                        detections.append(d)
                        existing.append(d.bbox)
                parked_count = len(detections)

            # 车流量统计
            events: list[FlowEvent] = []
            if self.flow_counter is not None and tracks:
                events = self.flow_counter.update(tracks, timestamp)
                total_cross_count += len(events)
                for line_id, cnt in self.flow_counter.get_flow_stats().items():
                    total_flow_by_line[line_id] = cnt

            parked_counts.append(parked_count)
            # 不再累加每帧检测数，改为从追踪器获取唯一总数
            flow_timeline.append((timestamp, len(events)))

            elapsed = time.time() - t0
            fps_times.append(elapsed)
            fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0

            if config.enable_visualization:
                annotated = self._draw_frame(frame, tracks, detections, fps, parked_count, total_cross_count)
                cv2.imshow("Campus Bike Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1
            total_frames += 1

        if config.enable_visualization:
            cv2.destroyAllWindows()

        return self._build_report(session_id, start_time, time.time(),
                                  total_frames, parked_counts, total_flow_by_line,
                                  flow_timeline, self.tracker.get_total_unique_count())

    def _build_report(self, session_id, start_time, end_time,
                      total_frames, parked_counts, total_flow_by_line,
                      flow_timeline, total_detected_count) -> SessionReport:
        if not parked_counts:
            return SessionReport(session_id=session_id, start_time=start_time, end_time=end_time,
                                 total_frames=0, peak_parked_count=0, avg_parked_count=0.0,
                                 total_detected_count=0, total_flow_by_line={}, flow_timeline=[])
        return SessionReport(
            session_id=session_id, start_time=start_time, end_time=end_time,
            total_frames=total_frames,
            peak_parked_count=max(parked_counts),
            avg_parked_count=sum(parked_counts) / total_frames,
            total_detected_count=total_detected_count,
            total_flow_by_line=total_flow_by_line,
            flow_timeline=flow_timeline,
        )

    def _draw_frame(self, frame, tracks, detections, fps, parked_count, total_cross) -> np.ndarray:
        h, w = frame.shape[:2]
        out = frame.copy()

        # 只画追踪框（有 track ID 的），避免与检测框重叠造成混乱
        for track in tracks:
            x1, y1, x2, y2 = (int(track.bbox[0]*w), int(track.bbox[1]*h),
                               int(track.bbox[2]*w), int(track.bbox[3]*h))
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{track.confidence:.2f}" if hasattr(track, 'confidence') and track.confidence > 0 else ""
            if label:
                cv2.putText(out, label, (x1, max(y1-6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
            pts = track.trajectory[-20:]
            for i in range(1, len(pts)):
                p1 = (int(pts[i-1][0]*w), int(pts[i-1][1]*h))
                p2 = (int(pts[i][0]*w), int(pts[i][1]*h))
                cv2.line(out, p1, p2, (0, 200, 100), 2)

        # 计数线
        for line in self.config.count_lines:
            pt1 = (int(line.start[0]*w), int(line.start[1]*h))
            pt2 = (int(line.end[0]*w), int(line.end[1]*h))
            cv2.line(out, pt1, pt2, (0, 0, 255), 2)
            cv2.putText(out, line.line_id, (pt1[0]+4, pt1[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # 信息面板（显示当前帧检测数 + 穿越总数）
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (230, 128), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
        cv2.putText(out, f"FPS  : {fps:5.1f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"Now  : {parked_count:4d}", (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(out, f"Cross: {total_cross:4d}", (8, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 1, cv2.LINE_AA)
        return out


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / ((ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter)
