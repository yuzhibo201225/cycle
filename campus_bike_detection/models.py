"""核心数据模型定义"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Detection:
    """单帧目标检测结果"""
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) 归一化坐标
    confidence: float
    class_id: int  # 0 = bicycle
    frame_id: int


@dataclass
class Track:
    """跨帧目标轨迹"""
    track_id: int
    bbox: tuple[float, float, float, float]
    state: str
    age: int
    confidence: float = 0.0
    trajectory: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class CountLine:
    """虚拟计数线配置"""
    line_id: str
    start: tuple[float, float]  # 归一化坐标
    end: tuple[float, float]
    direction: str  # "both" | "up_to_down" | "left_to_right"


@dataclass
class FlowEvent:
    """车辆穿越计数线事件"""
    track_id: int
    line_id: str
    timestamp: float
    direction: str  # "forward" | "backward"


@dataclass
class FrameResult:
    """帧级统计结果"""
    frame_id: int
    timestamp: float
    detections: list[Detection]
    tracks: list[Track]
    flow_events: list[FlowEvent]
    parked_count: int               # 当前停放数量
    flow_stats: dict[str, int]      # 各线路车流量


@dataclass
class SessionReport:
    """会话级汇总报告"""
    session_id: str
    start_time: float
    end_time: float
    total_frames: int
    peak_parked_count: int
    avg_parked_count: float
    total_detected_count: int       # 整个会话出现过的唯一自行车总数（track ID 去重）
    total_flow_by_line: dict[str, int]
    flow_timeline: list[tuple[float, int]]


@dataclass
class SystemConfig:
    """系统配置"""
    source: str | int               # 输入源：摄像头 ID 或视频文件路径
    model_path: str                 # 模型路径
    mode: str                       # "static" | "dynamic" | "both"
    device: str                     # "cpu" | "cuda" | "npu"
    target_fps: int                 # 目标处理帧率
    enable_visualization: bool      # 是否输出标注帧
    count_lines: list[CountLine] = field(default_factory=list)  # 动态模式计数线配置
