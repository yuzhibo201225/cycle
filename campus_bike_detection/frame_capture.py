"""帧采集与预处理模块"""
from __future__ import annotations

import cv2
import numpy as np


class FrameCapture:
    """统一封装摄像头流和视频文件两种输入，输出标准化帧。"""

    def __init__(self, source: str | int, target_size: tuple[int, int] = (640, 640)) -> None:
        """
        source: 摄像头设备ID (int) 或视频文件路径 (str)
        target_size: YOLO输入尺寸 (width, height)
        """
        self.source = source
        self.target_size = target_size
        self._cap = cv2.VideoCapture(source)

    def read_frame(self) -> tuple[bool, np.ndarray]:
        """读取下一帧，返回 (success, bgr_frame)；流结束时返回 (False, None)。"""
        ret, frame = self._cap.read()
        if not ret:
            return False, None
        return True, frame

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize 到 target_size，归一化到 [0, 1]，返回 float32 张量 (H, W, 3)。"""
        w, h = self.target_size
        resized = cv2.resize(frame, (w, h))
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def release(self) -> None:
        """释放 cv2.VideoCapture 资源。"""
        self._cap.release()
