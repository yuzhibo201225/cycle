"""
设备自适应模块：根据设备算力配置选择最优轻量化模型。
"""

from dataclasses import dataclass


@dataclass
class DeviceProfile:
    has_npu: bool = False
    has_gpu: bool = False
    ram_mb: int = 512
    cpu_cores: int = 4


def select_lightweight_model(device_profile: dict) -> dict:
    """
    根据设备算力配置选择最优轻量化模型。

    四级策略：
    1. has_npu → YOLOv8n-INT8 / tflite / npu
    2. has_gpu AND ram_mb >= 2048 → YOLOv8s-FP16 / onnx / cuda
    3. ram_mb >= 1024 → YOLOv8n-INT8 / ncnn / cpu
    4. else → YOLOv8n-distilled / ncnn / cpu (input_size=320)

    Args:
        device_profile: 设备描述字典，包含 has_npu, has_gpu, ram_mb, cpu_cores（可选）

    Returns:
        模型配置字典，包含 model, format, backend, input_size
    """
    has_npu = device_profile.get("has_npu", False)
    has_gpu = device_profile.get("has_gpu", False)
    ram_mb = device_profile.get("ram_mb", 0)

    if has_npu:
        return {
            "model": "YOLOv8n-INT8",
            "format": "tflite",
            "backend": "npu",
            "input_size": 640,
        }
    elif has_gpu and ram_mb >= 2048:
        return {
            "model": "YOLOv8s-FP16",
            "format": "onnx",
            "backend": "cuda",
            "input_size": 640,
        }
    elif ram_mb >= 1024:
        return {
            "model": "YOLOv8n-INT8",
            "format": "ncnn",
            "backend": "cpu",
            "input_size": 640,
        }
    else:
        return {
            "model": "YOLOv8n-distilled",
            "format": "ncnn",
            "backend": "cpu",
            "input_size": 320,
        }
