"""YOLO 推理引擎：支持 .pt / .onnx / .tflite 格式"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from campus_bike_detection.models import Detection


class YOLOInferenceEngine:
    """轻量化 YOLO 推理引擎，支持多种模型格式。"""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        conf_thresh: float = 0.4,
        expected_hash: str | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        if expected_hash is not None:
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            if sha256 != expected_hash:
                raise ValueError(
                    f"模型文件 SHA256 不匹配: 期望 {expected_hash}, 实际 {sha256}"
                )

        self.model_path = model_path
        self.device = device
        self.conf_thresh = conf_thresh
        self._ext = path.suffix.lower()
        self._model = self._load_model(path)

    # ------------------------------------------------------------------
    # 内部：模型加载
    # ------------------------------------------------------------------

    def _load_model(self, path: Path):
        if self._ext == ".pt":
            import torch
            from ultralytics import YOLO  # type: ignore
            model = YOLO(str(path))
            # 检测 CUDA 是否真正可用，不可用时自动回退 CPU
            if self.device == "cuda" and not torch.cuda.is_available():
                print("[警告] CUDA 不可用，自动切换到 CPU 推理")
                self.device = "cpu"
            model.to(self.device)
            return model

        elif self._ext == ".onnx":
            import onnxruntime as ort  # type: ignore
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )
            return ort.InferenceSession(str(path), providers=providers)

        elif self._ext == ".tflite":
            try:
                import tflite_runtime.interpreter as tflite  # type: ignore
                interpreter = tflite.Interpreter(model_path=str(path))
            except ImportError:
                import tensorflow as tf  # type: ignore
                interpreter = tf.lite.Interpreter(model_path=str(path))
            interpreter.allocate_tensors()
            return interpreter

        else:
            raise ValueError(f"不支持的模型格式: {self._ext}")

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def infer(self, tensor: np.ndarray, frame_id: int) -> list[Detection]:
        """执行推理，返回 class_id==0 且 confidence >= conf_thresh 的检测列表。"""
        if self._ext == ".pt":
            return self._infer_pt(tensor, frame_id)
        elif self._ext == ".onnx":
            return self._infer_onnx(tensor, frame_id)
        elif self._ext == ".tflite":
            return self._infer_tflite(tensor, frame_id)
        return []

    def _infer_pt(self, tensor: np.ndarray, frame_id: int) -> list[Detection]:
        results = self._model(tensor, verbose=False)
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                # COCO class 1 = bicycle（自行车）
                if cls != 1 or conf < self.conf_thresh:
                    continue
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                detections.append(
                    Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=conf,
                        class_id=cls,
                        frame_id=frame_id,
                    )
                )
        return detections

    def _infer_onnx(self, tensor: np.ndarray, frame_id: int) -> list[Detection]:
        session = self._model
        input_name = session.get_inputs()[0].name

        # 确保输入为 float32，shape [1, C, H, W] 或 [1, H, W, C]
        inp = tensor.astype(np.float32)
        if inp.ndim == 3:
            inp = np.expand_dims(inp, 0)

        outputs = session.run(None, {input_name: inp})
        detections: list[Detection] = []

        try:
            # 标准 YOLO ONNX 输出: [1, num_det, 6] 或 [1, 6, num_det]
            out = outputs[0]
            if out.ndim == 3:
                if out.shape[2] == 6:
                    # [batch, num_det, 6]: x1,y1,x2,y2,conf,cls
                    rows = out[0]
                elif out.shape[1] == 6:
                    # [batch, 6, num_det] → transpose
                    rows = out[0].T
                else:
                    rows = out[0]

                for row in rows:
                    if len(row) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls = (
                        float(row[0]), float(row[1]),
                        float(row[2]), float(row[3]),
                        float(row[4]), float(row[5]),
                    )
                    if int(cls) != 1 or conf < self.conf_thresh:
                        continue
                    detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_id=int(cls),
                            frame_id=frame_id,
                        )
                    )
        except Exception:
            pass

        return detections

    def _infer_tflite(self, tensor: np.ndarray, frame_id: int) -> list[Detection]:
        interpreter = self._model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        inp = tensor.astype(np.float32)
        if inp.ndim == 3:
            inp = np.expand_dims(inp, 0)

        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()

        detections: list[Detection] = []
        try:
            out = interpreter.get_tensor(output_details[0]["index"])
            # 假设输出 [1, num_det, 6]
            if out.ndim == 3:
                rows = out[0]
                for row in rows:
                    if len(row) < 6:
                        continue
                    x1, y1, x2, y2, conf, cls = (
                        float(row[0]), float(row[1]),
                        float(row[2]), float(row[3]),
                        float(row[4]), float(row[5]),
                    )
                    if int(cls) != 1 or conf < self.conf_thresh:
                        continue
                    detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_id=int(cls),
                            frame_id=frame_id,
                        )
                    )
        except Exception:
            pass

        return detections

    # ------------------------------------------------------------------
    # 预热
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """用全零 BGR 帧执行一次推理，消除首帧延迟。"""
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        if self._ext == ".pt":
            self._model(dummy, verbose=False, imgsz=320)
        else:
            self.infer(dummy, frame_id=-1)
